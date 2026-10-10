"""Business outputs a store owner can act on, each as a ready-to-download table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from . import models


def customer_value(lines: pd.DataFrame) -> pd.DataFrame:
    known = lines.dropna(subset=["customer_id"]).assign(revenue=lambda d: d["quantity"] * d["price"].fillna(0))
    return known.groupby("customer_id").agg(
        orders=("order_id", "nunique"), revenue=("revenue", "sum"), last_order=("date", "max"),
        first_order=("date", "min"))


def _reason(model, data: models.Interactions, row: int, item: int, names: pd.Series) -> str:
    own = data.X[row]
    if item in own.indices:
        orders = round(float(np.expm1(own[0, item])))
        return f"Re-order: bought in {orders} order{'s' if orders != 1 else ''} before"
    weights = getattr(model, "weights", None)
    if weights is None and hasattr(model, "personal"):
        weights = getattr(model.personal, "weights", None)
    if weights is None:
        weights = getattr(model, "similarity", None)
    if weights is None or own.nnz == 0:
        return "Popular with similar customers"
    contributions = own.data * weights[own.indices, item]
    best = own.indices[int(np.argmax(contributions))]
    return f"Because they bought {names.get(data.products[best], data.products[best])}"


def personal_picks(model, data: models.Interactions, names: pd.Series, k: int = 10,
                   discovery: bool = False, customers: list[str] | None = None) -> pd.DataFrame:
    rows = np.arange(len(data.customers)) if customers is None else data.customers.get_indexer(customers)
    rows = rows[rows >= 0]
    out = []
    for start in range(0, len(rows), 2000):
        r = rows[start:start + 2000]
        scores = model.scores(data.X[r])
        top = models.top_k(scores, k, exclude=data.X[r] if discovery else None)
        for i, row in enumerate(r):
            for rank, item in enumerate(top[i], 1):
                if not np.isfinite(scores[i, item]):
                    continue
                pid = data.products[item]
                out.append({"customer_id": data.customers[row], "rank": rank, "product_id": pid,
                            "product_name": names.get(pid, pid), "reason": _reason(model, data, row, item, names)})
    return pd.DataFrame(out)


def frequently_bought_together(lines: pd.DataFrame, names: pd.Series, min_orders: int = 10,
                               per_product: int = 5, min_lift: float = 1.5) -> pd.DataFrame:
    """Order-level co-purchase pairs (guest orders included), ranked by confidence.

    Ranking by lift alone surfaces rare oddities (bought together in 1% of orders, but rarely
    bought at all); a product page needs the companions people actually add, so pairs are ranked
    by confidence and lift is used as a filter (only pairs bought together well above chance).

    support    = share of all orders containing both products
    confidence = share of orders with A that also contain B
    lift       = confidence / share of orders with B  (above 1 means bought together more than by chance)
    """
    baskets = lines.drop_duplicates(["order_id", "product_id"])
    counts = baskets["product_id"].value_counts()
    keep = counts[counts >= min_orders].index
    baskets = baskets[baskets["product_id"].isin(keep)]
    orders = pd.Index(baskets["order_id"].unique())
    products = pd.Index(keep)
    M = sparse.csr_matrix((np.ones(len(baskets)), (orders.get_indexer(baskets["order_id"]),
                                                     products.get_indexer(baskets["product_id"]))),
                          shape=(len(orders), len(products)))
    co = (M.T @ M).tocoo()
    n_orders = len(orders)
    item_orders = np.asarray(M.sum(axis=0)).ravel()
    mask = (co.row != co.col) & (co.data >= min_orders)
    a, b, both = co.row[mask], co.col[mask], co.data[mask]
    confidence = both / item_orders[a]
    lift = confidence / (item_orders[b] / n_orders)
    pairs = pd.DataFrame({
        "product_id": products[a], "together_with_id": products[b], "orders_together": both.astype(int),
        "confidence": confidence, "lift": lift,
    })
    pairs = pairs[pairs["lift"] >= min_lift].sort_values(["product_id", "confidence"], ascending=[True, False])
    pairs = pairs.groupby("product_id").head(per_product)
    pairs.insert(1, "product_name", pairs["product_id"].map(names))
    pairs.insert(3, "together_with", pairs["together_with_id"].map(names))
    return pairs.reset_index(drop=True)


def bundles(pairs: pd.DataFrame, lines: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Bundle candidates: pairs bought together often (volume) and reliably (confidence ≥ 30%)."""
    if pairs.empty:
        return pairs
    p = pairs[pairs["confidence"] >= 0.3].copy()
    key = p.apply(lambda r: tuple(sorted((r["product_id"], r["together_with_id"]))), axis=1)
    p = p.assign(pair=key).sort_values("orders_together", ascending=False).drop_duplicates("pair")
    prices = lines.groupby("product_id")["price"].median()
    p["bundle_price_before_discount"] = (p["product_id"].map(prices) + p["together_with_id"].map(prices)).round(2)
    return p.head(top)[["product_name", "together_with", "orders_together", "confidence", "lift",
                        "bundle_price_before_discount"]].reset_index(drop=True)


def campaign_audience(model, data: models.Interactions, product_id: str, values: pd.DataFrame,
                      top: int = 200) -> pd.DataFrame:
    """Customers most likely to buy `product_id` who have not bought it yet."""
    item = data.products.get_loc(product_id)
    scores = np.concatenate([model.scores(data.X[s:s + 2000])[:, item]
                             for s in range(0, len(data.customers), 2000)])
    already = np.asarray(data.X[:, item].todense()).ravel() > 0
    scores[already] = -np.inf
    order = np.argsort(-scores)[:top]
    out = pd.DataFrame({"customer_id": data.customers[order], "score": scores[order]})
    out = out[np.isfinite(out["score"])]
    out["likelihood_rank"] = np.arange(1, len(out) + 1)
    return out.merge(values[["orders", "revenue", "last_order"]], left_on="customer_id", right_index=True,
                     how="left").drop(columns="score")


def win_back(model, data: models.Interactions, values: pd.DataFrame, names: pd.Series,
             inactive_days: int = 90, top: int = 200) -> pd.DataFrame:
    """Valuable customers who stopped buying, each with three suggested products."""
    last = values["last_order"]
    reference = last.max()
    lapsed = values[(reference - last).dt.days >= inactive_days].sort_values("revenue", ascending=False).head(top)
    lapsed = lapsed[lapsed.index.isin(data.customers)]
    if lapsed.empty:
        return pd.DataFrame()
    picks = personal_picks(model, data, names, k=3, customers=list(lapsed.index))
    joined = picks.groupby("customer_id")["product_name"].agg(" | ".join).rename("suggested_products")
    out = lapsed.assign(days_since_last_order=(reference - lapsed["last_order"]).dt.days).join(joined)
    return out.reset_index()[["customer_id", "orders", "revenue", "days_since_last_order", "suggested_products"]]
