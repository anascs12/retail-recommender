"""Read a store's order export and map it to one standard schema.

Standard columns: order_id, customer_id, product_id, product_name, quantity, price, date.
customer_id and price are optional; everything else is required.

Supported out of the box: Shopify order exports, common WooCommerce order exports,
the UCI Online Retail format, and any CSV/Excel file whose headers resemble the
synonyms below (the user can override any mapping in the app).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

STANDARD = ["order_id", "customer_id", "product_id", "product_name", "quantity", "price", "date"]
REQUIRED = ["order_id", "product_name", "quantity", "date"]

# Header synonyms, most specific first. Matching ignores case, spaces, underscores and punctuation.
SYNONYMS: dict[str, list[str]] = {
    "order_id": ["order id", "order number", "order no", "invoice", "invoice no", "invoice number",
                 "order", "transaction id", "receipt", "bill no"],
    "customer_id": ["customer id", "email", "customer email", "billing email", "billing email address",
                    "customer", "customer name", "client id", "user id", "phone", "billing phone"],
    "product_id": ["lineitem sku", "sku", "stockcode", "stock code", "product id", "item id", "item code",
                   "product code", "barcode", "variant sku"],
    "product_name": ["lineitem name", "description", "product name", "item name", "product", "item", "title",
                     "product title"],
    "quantity": ["lineitem quantity", "quantity", "qty", "units", "item quantity", "quantity ordered"],
    "price": ["lineitem price", "price", "unit price", "item price", "item cost", "rate", "sale price"],
    "date": ["created at", "invoicedate", "invoice date", "order date", "date", "paid at", "date created",
             "transaction date", "order created"],
}


# Exact mappings for known export formats (used when the format is recognised).
PRESETS: dict[str, dict[str, str]] = {
    "Shopify": {"order_id": "Name", "customer_id": "Email", "product_id": "Lineitem sku",
                "product_name": "Lineitem name", "quantity": "Lineitem quantity", "price": "Lineitem price",
                "date": "Created at"},
    "UCI Online Retail": {"order_id": "Invoice", "customer_id": "Customer ID", "product_id": "StockCode",
                          "product_name": "Description", "quantity": "Quantity", "price": "Price",
                          "date": "InvoiceDate"},
}


def normalise(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


@dataclass
class Mapping:
    columns: dict[str, str | None]          # standard name -> source column
    source: str                              # detected format, e.g. "Shopify"
    notes: list[str] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        return [c for c in REQUIRED if not self.columns.get(c)]


def detect_format(headers: list[str]) -> str:
    h = {normalise(c) for c in headers}
    if {"lineitemname", "lineitemquantity", "name"} <= h:
        return "Shopify"
    if {"invoice", "stockcode", "customerid"} <= h:
        return "UCI Online Retail"
    if any("billing" in c for c in h) and any("order" in c for c in h):
        return "WooCommerce"
    return "Generic"


def suggest_mapping(headers: list[str]) -> Mapping:
    """Pick the best source column for each standard field (each source column used once)."""
    source_format = detect_format(headers)
    if source_format in PRESETS:
        by_norm_all = {normalise(c): c for c in headers}
        preset = {t: by_norm_all.get(normalise(c)) for t, c in PRESETS[source_format].items()}
        return Mapping(columns=preset, source=source_format)
    by_norm = {normalise(c): c for c in headers}
    used: set[str] = set()
    columns: dict[str, str | None] = {}
    for target in STANDARD:
        choice = None
        for synonym in SYNONYMS[target]:
            source = by_norm.get(normalise(synonym))
            if source and source not in used:
                choice = source
                break
        if choice is None:  # fall back to "contains" matching, e.g. "Order Date (UTC)"
            for synonym in SYNONYMS[target]:
                key = normalise(synonym)
                hits = [c for n, c in by_norm.items() if key and key in n and c not in used]
                if hits:
                    choice = hits[0]
                    break
        columns[target] = choice
        if choice:
            used.add(choice)
    return Mapping(columns=columns, source=source_format)


def read_file(name: str, data: bytes) -> pd.DataFrame:
    """Read CSV or Excel bytes (all sheets of an Excel file are stacked)."""
    if name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
        return pd.concat(sheets.values(), ignore_index=True)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file as CSV or Excel")


def apply_mapping(raw: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """Rename to the standard schema and parse types. Raises if required fields are missing."""
    if mapping.missing_required:
        raise ValueError(f"Missing required columns: {', '.join(mapping.missing_required)}")
    out = pd.DataFrame({t: raw[s] if s else pd.NA for t, s in mapping.columns.items()})

    # Shopify (and some other exports) only fill order-level fields on an order's first line.
    for col in ["customer_id", "date"]:
        if out[col].notna().any():
            out[col] = out.groupby("order_id")[col].transform("first")  # first non-empty value per order

    out["order_id"] = out["order_id"].astype(str).str.strip()
    out["product_name"] = out["product_name"].astype(str).str.strip()
    if out["product_id"].isna().all():
        out["product_id"] = out["product_name"]
        mapping.notes.append("No SKU column found: products are identified by name.")
    # SKUs are compared case-insensitively ("85049g" and "85049G" are the same product).
    out["product_id"] = out["product_id"].fillna(out["product_name"]).astype(str).str.strip().str.upper()
    out["customer_id"] = (out["customer_id"].astype("string").str.strip().str.lower()
                          .str.replace(r"^(\d+)\.0$", r"\1", regex=True))  # Excel turns 12448 into 12448.0
    out.loc[out["customer_id"].isin(["", "nan", "none", "<na>"]), "customer_id"] = pd.NA
    out["quantity"] = pd.to_numeric(out["quantity"].astype(str).str.replace(",", ""), errors="coerce")
    price = out["price"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    out["price"] = pd.to_numeric(price, errors="coerce")
    out["date"] = parse_dates(out["date"])
    return out


def is_day_first(text: pd.Series) -> bool:
    """For slash/dot/dash dates like 03/04/2025, decide day-first vs month-first from the data:
    any first number above 12 proves day-first; any second number above 12 proves month-first.
    Ambiguous files default to day-first, the convention in Pakistan, the UK and most of the world."""
    parts = text.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}").dropna().astype(int)
    if parts.empty:
        return False  # ISO dates (2025-04-03) are unambiguous
    month_first_proven = (parts[1] > 12).any() and not (parts[0] > 12).any()
    return not month_first_proven


def parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\s[+-]\d{4}$", "", regex=True)  # Shopify adds "+0500"
    return pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=is_day_first(text))
