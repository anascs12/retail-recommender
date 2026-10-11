"""BasketWise web app.  Run:  uv run streamlit run app/streamlit_app.py"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException

from basketwise import catalog, engine, importer, outputs, remote, similar

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo_store.parquet"

st.set_page_config(page_title="BasketWise · Product recommendations for your store", page_icon="🧺", layout="wide")


# ------------------------------------------------------------------ helpers
def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel opens Urdu/UTF-8 text correctly


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            frame = df.copy()
            for col in frame.select_dtypes(include=["datetimetz", "datetime"]).columns:
                frame[col] = frame[col].dt.tz_localize(None) if hasattr(frame[col].dt, "tz") else frame[col]
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


@st.cache_resource(show_spinner=False, max_entries=3)
def analyse(key: str, _lines: pd.DataFrame) -> engine.StoreResults:
    return engine.run(_lines)


@st.cache_data(show_spinner=False, ttl=600, max_entries=5)
def fetch_link(url: str) -> remote.Fetched:
    return remote.fetch(url)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=5)
def import_site(url: str, limit: int) -> catalog.Catalog:
    return catalog.import_catalog(url, limit=limit)


def load_demo() -> tuple[str, pd.DataFrame]:
    lines = pd.read_parquet(DEMO)
    return "demo-store-v1", lines


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧺 BasketWise")
    st.caption("Turn your order history into product recommendations your customers actually want.")
    source = st.radio("Data", ["Try the demo store", "Upload my orders", "Paste a link", "Import from my website"],
                      index=0)
    st.divider()
    st.markdown("**Works with** Shopify and WooCommerce order exports, POS exports, or any CSV/Excel "
                "with order, product, quantity and date columns.")
    st.markdown("🔒 **Privacy:** your file is processed in memory for this session only. "
                "Nothing is stored or shared.")

st.title("Product recommendations for your store")

# ------------------------------------------------------------------ website catalogue
if source == "Import from my website":
    st.subheader("Import your product catalogue from your website")
    st.markdown("Works with **Shopify** and **WooCommerce** stores, and most other shops that publish product "
                "pages for Google. Your **orders and customers are private** and are never read from the website; "
                "use *Upload my orders* or *Paste a link* for those.")
    site = st.text_input("Your store's website", placeholder="https://yourshop.pk")
    limit = st.select_slider("Maximum products to import", [100, 250, 500, 1000, 2500, 5000], value=500)
    st.caption("BasketWise respects your site's robots.txt and loads pages slowly (about one per second), so large "
               "catalogues on sites without a Shopify/WooCommerce feed can take a few minutes.")
    if not site:
        st.stop()
    try:
        with st.spinner("Reading your catalogue…"):
            cat = import_site(site, limit)
    except catalog.CatalogError as exc:
        st.error(str(exc))
        st.stop()
    except requests.RequestException:
        st.error("The website didn't respond. Check the address and try again.")
        st.stop()
    products = cat.products
    st.success(f"Imported **{len(products):,} products** from {cat.base_url} via **{cat.platform}**.")
    for note in cat.notes:
        st.caption(f"ℹ️ {note}")
    pairs = similar.similar_products(products)
    tab_sim, tab_cat = st.tabs(["🧩 Similar products", "📦 Catalogue"])
    with tab_sim:
        st.caption("Products that look alike by name, category, tags and description, in a similar price range. "
                   "Works for new products with no sales yet. Use these for 'You may also like' on product pages.")
        if pairs.empty:
            st.info("Not enough product text to find similar items.")
        else:
            chosen_title = st.selectbox("Product", sorted(pairs["title"].unique()))
            view = pairs[pairs["title"] == chosen_title][["similar_title", "similarity", "similar_url"]]
            st.dataframe(view, width="stretch", hide_index=True,
                         column_config={"similar_url": st.column_config.LinkColumn("Link"),
                                        "similarity": st.column_config.ProgressColumn("Similarity", min_value=0,
                                                                                      max_value=1, format="%.2f")})
            st.download_button("Download all similar-product pairs (CSV)", to_csv(pairs), "similar_products.csv",
                               "text/csv")
    with tab_cat:
        st.dataframe(products.drop(columns=["image"]), width="stretch", hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("Link")})
        st.download_button("Download catalogue (CSV)", to_csv(products), "catalogue.csv", "text/csv")
    st.info("**Next step:** add your order history (*Upload my orders* or *Paste a link*) to unlock "
            "frequently-bought-together, bundles and personal picks.")
    st.stop()

# ------------------------------------------------------------------ data in
if source == "Try the demo store":
    key, lines = load_demo()
    st.info("Demo: 12 months of orders from a real UK online gift shop (UCI Online Retail II, CC BY 4.0). "
            "Upload your own export or paste a link from the sidebar to see results for your store.")
else:
    if source == "Paste a link":
        link = st.text_input("Link to your orders file",
                             placeholder="https://docs.google.com/spreadsheets/d/…  or a Drive, Dropbox, OneDrive or direct CSV/Excel link")
        st.caption("Works with Google Sheets, Google Drive, Dropbox, OneDrive and direct CSV/Excel links. "
                   "Set sharing to **Anyone with the link can view**. A store's website address alone can't be used, "
                   "because orders are private.")
        if not link:
            st.stop()
        try:
            with st.spinner("Downloading your file…"):
                fetched = fetch_link(link)
        except remote.LinkError as exc:
            st.error(str(exc))
            st.stop()
        except requests.RequestException:
            st.error("Couldn't download the file (the site didn't respond). Check the link and try again.")
            st.stop()
        file_name, data = fetched.name, fetched.data
        st.caption(f"Downloaded {len(data) / 1024:,.0f} KB from {fetched.source}.")
    else:
        upload = st.file_uploader("Upload your orders export (CSV or Excel)", type=["csv", "xlsx", "xls"])
        if upload is None:
            st.markdown("""
**How to export your orders**
- **Shopify:** Orders → Export → *All orders* → *CSV for Excel*.
- **WooCommerce:** WooCommerce → Orders → Export (or an order-export plugin), including line items.
- **Other / POS:** any file with one row per product per order: order number, product name or SKU,
  quantity, date, and ideally customer email or phone.
""")
            st.stop()
        file_name, data = upload.name, upload.getvalue()
    try:
        raw = importer.read_file(file_name, data)
    except (ValueError, OSError, zipfile.BadZipFile, InvalidFileException):  # ParserError is a ValueError
        st.error("Couldn't read the file as CSV or Excel. Please check it's an orders export.")
        st.stop()
    suggestion = importer.suggest_mapping(list(raw.columns))
    st.success(f"Read {len(raw):,} rows · detected format: **{suggestion.source}**")
    with st.expander("Check how your columns were matched", expanded=bool(suggestion.missing_required)):
        options = ["— not in my file —", *raw.columns]
        chosen = {}
        cols = st.columns(4)
        for i, target in enumerate(importer.STANDARD):
            current = suggestion.columns.get(target)
            label = target.replace("_", " ") + (" *" if target in importer.REQUIRED else "")
            pick = cols[i % 4].selectbox(label, options, index=options.index(current) if current in options else 0)
            chosen[target] = None if pick == options[0] else pick
        st.dataframe(raw.head(5), width="stretch")
    mapping = importer.Mapping(columns=chosen, source=suggestion.source)
    if mapping.missing_required:
        st.error(f"Please choose a column for: {', '.join(mapping.missing_required)}")
        st.stop()
    try:
        lines = importer.apply_mapping(raw, mapping)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    for note in mapping.notes:
        st.caption(f"ℹ️ {note}")
    key = hashlib.sha256(data).hexdigest() + str(sorted(chosen.items()))

with st.spinner("Cleaning your data, testing models on your own history and building recommendations…"):
    results = analyse(key, lines)

s = results.summary
for w in results.warnings:
    st.warning(w)

# ------------------------------------------------------------------ results
tab_overview, tab_fbt, tab_bundles, tab_picks, tab_campaign, tab_winback = st.tabs(
    ["📊 Overview", "🛒 Bought together", "🎁 Bundles", "👤 Personal picks", "📣 Campaign lists", "🔁 Win-back"])

with tab_overview:
    c = st.columns(5)
    c[0].metric("Orders", f"{s['orders']:,}")
    c[1].metric("Products", f"{s['products']:,}")
    c[2].metric("Customers", f"{s['customers']:,}")
    c[3].metric("Repeat customers", f"{s['repeat_customer_share']:.0%}")
    first, last = pd.Timestamp(s["first_date"]), pd.Timestamp(s["last_date"])
    c[4].metric("Period", f"{first:%b %Y} – {last:%b %Y}")
    if results.proof:
        st.success("**Does it work on your data?**\n\n" + "\n\n".join(results.proof))
    if results.evaluation is not None:
        disc = results.evaluation[results.evaluation["mode"] == "Discovery"]
        disc = disc[disc["model"] != "Buy again"]
        fig = px.bar(disc.sort_values("hit_rate"), x="hit_rate", y="model", orientation="h", text_auto=".0%",
                     labels={"hit_rate": "Customers with at least one correct new product in their top 10", "model": ""},
                     title="Tested on your own recent orders: which method predicts new purchases best?",
                     template="plotly_white")
        fig.update_layout(height=300, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Full evaluation table"):
            st.dataframe(results.evaluation.round(3), width="stretch")
    removed = results.cleaning.removed
    if removed:
        st.caption("**Data cleaning:** " + " · ".join(f"{v:,} {k}" for k, v in removed.items())
                   + f". {results.cleaning.rows_out:,} of {results.cleaning.rows_in:,} rows used.")

with tab_fbt:
    st.subheader("Frequently bought together")
    st.caption("For each product, what customers most often buy with it, only where the pairing is well above "
               "chance (lift ≥ 1.5). Use these for 'Customers also bought' on product pages.")
    fbt = results.fbt
    if fbt.empty:
        st.info("Not enough multi-product orders yet.")
    else:
        by_sales = fbt.groupby("product_name")["orders_together"].sum().sort_values(ascending=False).index
        product = st.selectbox("Product (best-sellers first)", list(by_sales))
        view = fbt[fbt["product_name"] == product][["together_with", "orders_together", "confidence", "lift"]]
        st.dataframe(view.rename(columns={"together_with": "Often bought with", "orders_together": "Orders together",
                                          "confidence": "Share of this product's orders", "lift": "Lift"})
                     .style.format({"Share of this product's orders": "{:.0%}", "Lift": "{:.1f}"}),
                     width="stretch", hide_index=True)
        st.download_button("Download all pairs (CSV)", to_csv(fbt), "frequently_bought_together.csv", "text/csv")

with tab_bundles:
    st.subheader("Bundle ideas")
    st.caption("Pairs bought together often and reliably (in at least 30% of the first product's orders).")
    st.dataframe(results.bundles.style.format({"confidence": "{:.0%}", "lift": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.download_button("Download bundles (CSV)", to_csv(results.bundles), "bundle_ideas.csv", "text/csv")

if results.data is None:
    for tab in (tab_picks, tab_campaign, tab_winback):
        with tab:
            st.info("Personal recommendations need more identified customers (see the note above).")
    st.stop()

with tab_picks:
    st.subheader("Personal picks")
    discovery = st.toggle("Only products the customer hasn't bought yet", value=True)
    model = results.model_discovery if discovery else results.model_everything
    top_customers = results.values.sort_values("revenue", ascending=False).index[:500]
    customer = st.selectbox("Customer (top 500 by spend)", [c for c in top_customers if c in results.data.customers])
    picks = outputs.personal_picks(model, results.data, results.names, k=10, discovery=discovery, customers=[customer])
    st.dataframe(picks[["rank", "product_name", "reason"]], width="stretch", hide_index=True)
    if st.button("Prepare picks for all customers"):
        with st.spinner("Scoring every customer…"):
            everyone = outputs.personal_picks(model, results.data, results.names, k=5, discovery=discovery)
        st.download_button("Download all picks (CSV)", to_csv(everyone), "personal_picks.csv", "text/csv")

with tab_campaign:
    st.subheader("Who to target for a product")
    st.caption("Customers most likely to buy a product they haven't bought yet: an email or WhatsApp list.")
    popular = results.fbt["product_name"].value_counts().index.tolist() or list(results.names.values)
    name_to_id = {v: k for k, v in results.names.items() if k in set(results.data.products)}
    product_name = st.selectbox("Product to promote", [p for p in popular if p in name_to_id] or sorted(name_to_id))
    size = st.slider("List size", 50, 1000, 200, step=50)
    audience = outputs.campaign_audience(results.model_discovery, results.data, name_to_id[product_name],
                                         results.values, top=size)
    st.dataframe(audience, width="stretch", hide_index=True)
    st.download_button("Download list (CSV)", to_csv(audience), "campaign_list.csv", "text/csv")

with tab_winback:
    st.subheader("Win back lapsed customers")
    days = st.slider("Inactive for at least (days)", 30, 365, 90, step=15)
    wb = outputs.win_back(results.model_discovery, results.data, results.values, results.names, inactive_days=days)
    if wb.empty:
        st.info("No lapsed customers for this setting.")
    else:
        st.caption(f"{len(wb)} valuable customers, each with three products they're likely to want.")
        st.dataframe(wb, width="stretch", hide_index=True)
        st.download_button("Download win-back list (CSV)", to_csv(wb), "win_back.csv", "text/csv")
        st.download_button("Download everything (Excel)",
                           to_excel({"Win-back": wb, "Bundles": results.bundles, "Bought together": results.fbt}),
                           "basketwise_results.xlsx")
