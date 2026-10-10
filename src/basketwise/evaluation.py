"""Honest offline evaluation: learn from older orders, check against what customers bought next."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from . import models


@dataclass
class Split:
    train_lines: pd.DataFrame
    data: models.Interactions
    test_rows: np.ndarray                 # row indices (into data.customers) of evaluated customers
    truth_all: list[set[int]]             # products bought in the test window
    truth_new: list[set[int]]             # ... excluding products the customer had bought before
    cutoff: pd.Timestamp
    test_days: int


def temporal_split(lines: pd.DataFrame, test_days: int | None = None) -> Split:
    """Hold out the most recent period: 90 days, or a fifth of the history for short histories."""
    span = (lines["date"].max() - lines["date"].min()).days
    test_days = test_days or int(min(90, max(14, span // 5)))
    cutoff = lines["date"].max() - pd.Timedelta(days=test_days)
    train = lines[lines["date"] <= cutoff]
    test = lines[(lines["date"] > cutoff)].dropna(subset=["customer_id"])
    data = models.build_interactions(train)
    test = test[test["customer_id"].isin(data.customers) & test["product_id"].isin(data.products)]
    grouped = test.groupby("customer_id")["product_id"].agg(set)
    rows = data.customers.get_indexer(grouped.index)
    truth_all = [set(data.products.get_indexer(list(s))) for s in grouped]
    seen = [set(data.X[r].indices) for r in rows]
    truth_new = [t - s for t, s in zip(truth_all, seen, strict=True)]
    return Split(train, data, rows, truth_all, truth_new, cutoff, test_days)


def ranking_metrics(recommended: np.ndarray, truth: list[set[int]], k: int) -> dict[str, float]:
    hits, precision, recall, ndcg = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for recs, actual in zip(recommended, truth, strict=True):
        if not actual:
            continue
        rel = np.array([r in actual for r in recs[:k]], dtype=float)
        hits.append(float(rel.any()))
        precision.append(rel.sum() / k)
        recall.append(rel.sum() / len(actual))
        ideal = discounts[: min(len(actual), k)].sum()
        ndcg.append(float((rel * discounts).sum() / ideal))
    return {"hit_rate": float(np.mean(hits)), "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)), "ndcg": float(np.mean(ndcg)), "customers": len(hits)}


def evaluate(split: Split, candidates: list, k: int = 10, batch: int = 2000) -> pd.DataFrame:
    """Score each model in two modes:
    - 'Everything' : any product they bought next counts, including re-orders.
    - 'Discovery'  : only products new to the customer count, and already-bought ones are not recommended.
    """
    X = split.data.X
    rows_out = []
    for model in candidates:
        model.fit(split.data)
        recs_all, recs_new = [], []
        for start in range(0, len(split.test_rows), batch):
            r = split.test_rows[start:start + batch]
            s = model.scores(X[r])
            recs_all.append(models.top_k(s, k))
            recs_new.append(models.top_k(s, k, exclude=X[r]))
        recs_all, recs_new = np.vstack(recs_all), np.vstack(recs_new)
        n_items = X.shape[1]
        for mode, recs, truth in [("Everything", recs_all, split.truth_all), ("Discovery", recs_new, split.truth_new)]:
            m = ranking_metrics(recs, truth, k)
            m.update(model=model.name, mode=mode, coverage=len(np.unique(recs)) / n_items)
            rows_out.append(m)
    return pd.DataFrame(rows_out)[["mode", "model", "hit_rate", "precision", "recall", "ndcg", "coverage", "customers"]]


def as_sparse_rows(rows: list[list[int]], n_items: int) -> sparse.csr_matrix:
    data = [1.0 for r in rows for _ in r]
    indices = [i for r in rows for i in r]
    indptr = np.cumsum([0] + [len(r) for r in rows])
    return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), n_items))
