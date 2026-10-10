"""Clean standardised order lines and explain every removal to the store owner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Lines that are not real products: shipping, fees, discounts, manual adjustments, vouchers.
# Ambiguous words (carriage, postage, discount...) only count at the START of the line name, so
# real products such as "FRENCH CARRIAGE LANTERN" are kept. Unambiguous phrases match anywhere.
NON_PRODUCT_START = re.compile(
    r"^\s*(?:dotcom\s+)?(?:next day\s+)?(?:postage|shipping|delivery|carriage|freight|bank charges?|fees?|"
    r"discount|manual|adjust(?:ment)?|samples?|test)\b",
    re.IGNORECASE,
)
NON_PRODUCT_ANYWHERE = re.compile(
    r"gift ?(?:card|voucher)|test product|adjustment by|amazon fee|cod charges?|shipping (?:fee|charges?)|"
    r"delivery charges?|packing charges?",
    re.IGNORECASE,
)


def is_non_product_name(names: pd.Series) -> pd.Series:
    return names.str.contains(NON_PRODUCT_START, na=False) | names.str.contains(NON_PRODUCT_ANYWHERE, na=False)


NON_PRODUCT_CODE = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "D", "S", "B", "CRUK", "PADS",
                    "ADJUST", "ADJUST2", "TEST001", "TEST002"}


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int = 0
    removed: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int) -> None:
        if n:
            self.removed[reason] = self.removed.get(reason, 0) + int(n)


def clean(lines: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(lines))
    df = lines.copy()

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        report.add(reason, mask.sum())
        df = df[~mask]

    drop(df["date"].isna(), "unreadable date")
    drop(df["quantity"].isna(), "unreadable quantity")
    cancelled = df["order_id"].str.upper().str.startswith("C") & df["order_id"].str[1:].str.isdigit()
    drop(cancelled | (df["quantity"] <= 0), "cancellations and returns")
    drop(df["product_id"].str.upper().isin(NON_PRODUCT_CODE)
         | is_non_product_name(df["product_name"]), "shipping, fees and other non-product lines")
    drop(df["price"].notna() & (df["price"] <= 0), "zero-price lines (samples, damages, free items)")
    drop(df["product_name"].isin(["", "nan", "None"]) & df["product_id"].isin(["", "nan", "None"]),
         "lines without a product")

    # One canonical display name per product (the most frequent spelling).
    names = df.groupby("product_id")["product_name"].agg(lambda s: s.value_counts().index[0])
    df["product_name"] = df["product_id"].map(names)
    report.rows_out = len(df)
    return df.reset_index(drop=True), report


def summary(df: pd.DataFrame) -> dict:
    known = df["customer_id"].notna()
    return {
        "orders": int(df["order_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "customers": int(df.loc[known, "customer_id"].nunique()),
        "guest_order_share": round(float(1 - df.loc[known, "order_id"].nunique() / max(df["order_id"].nunique(), 1)), 3),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "repeat_customer_share": round(float(
            (df[known].groupby("customer_id")["order_id"].nunique() > 1).mean()), 3) if known.any() else 0.0,
    }
