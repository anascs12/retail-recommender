import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from basketwise import cleaning, engine, evaluation, importer, models, outputs

SHOPIFY_CSV = """Name,Email,Financial Status,Created at,Lineitem quantity,Lineitem name,Lineitem price,Lineitem sku
#1001,Ali@Example.com,paid,2025-03-04 10:15:00 +0500,2,Blue Mug,450.00,mug-blue
#1001,,,,1,Coaster Set,300.00,COASTER-1
#1002,sara@example.com,paid,2025-03-05 12:00:00 +0500,1,Shipping,200.00,
#1002,,,,1,Blue Mug,450.00,MUG-BLUE
"""


def test_shopify_export_is_detected_and_order_fields_are_filled_down():
    raw = importer.read_file("orders_export.csv", SHOPIFY_CSV.encode())
    mapping = importer.suggest_mapping(list(raw.columns))
    assert mapping.source == "Shopify" and mapping.columns["order_id"] == "Name"
    lines = importer.apply_mapping(raw, mapping)
    second_line = lines.iloc[1]
    assert second_line["customer_id"] == "ali@example.com"        # filled from the order's first line
    assert second_line["date"] == pd.Timestamp("2025-03-04 10:15")  # "+0500" suffix handled
    assert set(lines["product_id"]) >= {"MUG-BLUE"}               # SKUs compared case-insensitively
    assert lines.loc[lines["product_name"] == "Blue Mug", "product_id"].nunique() == 1


def test_generic_headers_are_matched_by_synonyms():
    headers = ["Order Number", "Billing Email", "Item Name", "SKU", "Qty", "Order Date (PKT)", "Unit Price"]
    mapping = importer.suggest_mapping(headers)
    assert mapping.columns == {"order_id": "Order Number", "customer_id": "Billing Email", "product_id": "SKU",
                               "product_name": "Item Name", "quantity": "Qty", "price": "Unit Price",
                               "date": "Order Date (PKT)"}


@pytest.mark.parametrize("dates, expected_month", [
    (["03/04/2025", "25/12/2025"], 4),   # a day above 12 proves day-first → 3 April
    (["03/04/2025", "12/25/2025"], 3),   # a second number above 12 proves month-first → 4 March
    (["03/04/2025"], 4),                 # ambiguous → day-first (Pakistan, UK)
    (["2025-03-04"], 3),                 # ISO is unambiguous
])
def test_date_order_is_detected_from_the_data(dates, expected_month):
    assert importer.parse_dates(pd.Series(dates)).iloc[0].month == expected_month


def test_missing_required_columns_are_reported():
    mapping = importer.suggest_mapping(["Customer", "Product"])
    assert "order_id" in mapping.missing_required and "date" in mapping.missing_required


def test_excel_style_numeric_customer_ids_are_normalised():
    raw = pd.DataFrame({"Invoice": ["1"], "StockCode": ["a1"], "Description": ["x"], "Quantity": ["1"],
                        "InvoiceDate": ["2025-01-01"], "Price": ["1"], "Customer ID": ["12448.0"]})
    lines = importer.apply_mapping(raw, importer.suggest_mapping(list(raw.columns)))
    assert lines["customer_id"].iloc[0] == "12448"


def standard(rows):
    return pd.DataFrame(rows, columns=["order_id", "customer_id", "product_id", "product_name", "quantity",
                                       "price", "date"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_cleaning_removes_non_products_but_keeps_real_products_with_similar_words():
    lines = standard([
        ["1", "a", "POST", "POSTAGE", 1, 18.0, "2025-01-01"],
        ["1", "a", "X1", "Next Day Carriage", 1, 30.0, "2025-01-01"],
        ["1", "a", "X2", "FRENCH CARRIAGE LANTERN", 1, 5.0, "2025-01-01"],
        ["C2", "a", "X3", "MUG", -1, 2.0, "2025-01-02"],
        ["3", "a", "X4", "Dotcomgiftshop Gift Voucher £20.00", 1, 20.0, "2025-01-03"],
        ["4", "a", "X5", "FREE SAMPLE MUG", 1, 0.0, "2025-01-04"],
    ])
    clean, report = cleaning.clean(lines)
    assert clean["product_name"].tolist() == ["FRENCH CARRIAGE LANTERN"]
    assert report.removed["cancellations and returns"] == 1
    assert report.rows_in == 6 and report.rows_out == 1


def matrix(rows):
    return sparse.csr_matrix(np.array(rows, dtype=float))


def toy_interactions():
    # products: 0=tea cup, 1=saucer (always together), 2=lunch bag (unrelated)
    X = matrix([[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
    return models.Interactions(X=X, customers=pd.Index(list("abcde")), products=pd.Index(["cup", "saucer", "bag"]),
                               last_purchase=pd.Series(dtype="datetime64[ns]"))


@pytest.mark.parametrize("model", [models.EASE(l2=1.0), models.ItemKNN(neighbours=2, shrink=0.0)])
def test_item_models_recommend_the_co_purchased_item(model):
    data = toy_interactions()
    model.fit(data)
    scores = model.scores(data.X[[4]])   # customer 'e' bought only the cup
    assert models.top_k(scores, 1, exclude=data.X[[4]])[0, 0] == 1  # → saucer, not the lunch bag


def test_top_k_never_returns_excluded_items():
    scores = np.array([[5.0, 4.0, 3.0, 2.0]])
    assert models.top_k(scores, 2, exclude=matrix([[1, 0, 1, 0]])).tolist() == [[1, 3]]


def test_ranking_metrics_on_a_known_case():
    m = evaluation.ranking_metrics(np.array([[3, 1, 2]]), [{1, 9}], k=3)
    assert m["hit_rate"] == 1.0 and m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5
    assert m["ndcg"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def synthetic_store(customers=80, days=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(customers):
        favourite = rng.integers(0, 5)
        for o in range(rng.integers(2, 6)):
            day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, days)))
            order = f"{c}-{o}"
            for p in {favourite, (favourite + 1) % 5, int(rng.integers(5, 20))}:
                rows.append([order, f"c{c}", f"P{p}", f"Product {p}", 1, 10.0, day])
    return standard(rows)


def test_temporal_split_never_leaks_future_orders_into_training():
    split = evaluation.temporal_split(synthetic_store())
    assert split.train_lines["date"].max() <= split.cutoff


def test_frequently_bought_together_lift_and_confidence():
    lines = standard([[str(o), None, p, p, 1, 1.0, "2025-01-01"] for o in range(10) for p in ("cup", "saucer")]
                     + [[str(o), None, "bag", "bag", 1, 1.0, "2025-01-01"] for o in range(10, 20)])
    names = pd.Series({"cup": "cup", "saucer": "saucer", "bag": "bag"})
    pairs = outputs.frequently_bought_together(lines, names, min_orders=3)
    cup = pairs[pairs["product_id"] == "cup"].iloc[0]
    assert cup["together_with"] == "saucer" and cup["confidence"] == 1.0 and cup["lift"] == pytest.approx(2.0)
    assert "bag" not in set(pairs["together_with"])


def test_engine_on_a_small_store_warns_instead_of_overpromising():
    results = engine.run(synthetic_store(customers=20))
    assert results.data is None and any("personal picks need" in w for w in results.warnings)


def test_engine_end_to_end_on_a_synthetic_store():
    results = engine.run(synthetic_store(customers=300, days=300))
    assert results.data is not None and results.chosen["Discovery"]
    picks = outputs.personal_picks(results.model_discovery, results.data, results.names, k=3, discovery=True,
                                   customers=[results.data.customers[0]])
    assert len(picks) == 3 and picks["reason"].str.len().gt(0).all()
