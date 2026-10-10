"""Recommendation models on a customer × product purchase matrix.

All models share one interface: fit(X) on a sparse matrix (rows = customers,
columns = products), then scores(rows) returns a dense score array for those
customers. Higher scores mean "more likely to buy next".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Interactions:
    X: sparse.csr_matrix                 # customer × product, weight = log(1 + number of orders)
    customers: pd.Index
    products: pd.Index
    last_purchase: pd.Series             # customer -> last order date in this window


def build_interactions(lines: pd.DataFrame, min_customers_per_product: int = 2) -> Interactions:
    """Known customers only; products bought by fewer than `min_customers_per_product` are dropped
    (nothing can be learned about them)."""
    known = lines.dropna(subset=["customer_id"])
    pairs = known.groupby(["customer_id", "product_id"])["order_id"].nunique().rename("orders").reset_index()
    reach = pairs.groupby("product_id")["customer_id"].nunique()
    pairs = pairs[pairs["product_id"].isin(reach[reach >= min_customers_per_product].index)]
    customers = pd.Index(sorted(pairs["customer_id"].unique()))
    products = pd.Index(sorted(pairs["product_id"].unique()))
    X = sparse.csr_matrix(
        (np.log1p(pairs["orders"].to_numpy(dtype=float)),
         (customers.get_indexer(pairs["customer_id"]), products.get_indexer(pairs["product_id"]))),
        shape=(len(customers), len(products)),
    )
    last = known.groupby("customer_id")["date"].max().reindex(customers)
    return Interactions(X=X, customers=customers, products=products, last_purchase=last)


class Popularity:
    """Most popular products recently (distinct customers in the last `days` days)."""

    name = "Popular now"

    def __init__(self, lines: pd.DataFrame, days: int = 90):
        self.lines, self.days = lines, days

    def fit(self, data: Interactions):
        recent = self.lines[self.lines["date"] >= self.lines["date"].max() - pd.Timedelta(days=self.days)]
        counts = recent.dropna(subset=["customer_id"]).groupby("product_id")["customer_id"].nunique()
        self.vector = counts.reindex(data.products).fillna(0).to_numpy(dtype=float)
        return self

    def scores(self, X_rows) -> np.ndarray:
        return np.tile(self.vector, (X_rows.shape[0], 1))


class BuyAgain:
    """Each customer's own most-purchased products (a strong baseline in retail)."""

    name = "Buy again"

    def fit(self, data: Interactions):
        return self

    def scores(self, X_rows) -> np.ndarray:
        return np.asarray(X_rows.todense())


class ItemKNN:
    """Item-to-item collaborative filtering: cosine similarity between products' buyer sets,
    with shrinkage so rarely co-bought pairs don't look artificially similar."""

    name = "Customers also bought (item-kNN)"

    def __init__(self, neighbours: int = 50, shrink: float = 10.0):
        self.neighbours, self.shrink = neighbours, shrink

    def fit(self, data: Interactions):
        B = (data.X > 0).astype(float).tocsc()
        co = (B.T @ B).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[0]:  # keep only each product's strongest neighbours
            cut = np.partition(sim, -self.neighbours, axis=1)[:, -self.neighbours][:, None]
            sim[sim < cut] = 0.0
        self.similarity = sim
        return self

    def scores(self, X_rows) -> np.ndarray:
        return np.asarray(X_rows @ self.similarity)


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019): a closed-form item-to-item model that is
    a strong, simple baseline in recommender research. One matrix inversion; λ controls smoothing."""

    name = "EASE"

    def __init__(self, l2: float = 300.0):
        self.l2 = l2

    def fit(self, data: Interactions):
        G = (data.X.T @ data.X).toarray()
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.weights = B
        return self

    def scores(self, X_rows) -> np.ndarray:
        return np.asarray(X_rows @ self.weights)


class Hybrid:
    """Blend of a personal model and each customer's own history: a small boost for products the
    customer already buys, because repeat purchases are common in retail."""

    def __init__(self, personal, repeat_weight: float = 0.5):
        self.personal, self.repeat_weight = personal, repeat_weight
        self.name = f"{personal.name} + buy again"

    def fit(self, data: Interactions):
        self.personal.fit(data)
        return self

    def scores(self, X_rows) -> np.ndarray:
        p = self.personal.scores(X_rows)
        own = np.asarray(X_rows.todense())
        scale = lambda a: a / (np.abs(a).max(axis=1, keepdims=True) + 1e-12)
        return (1 - self.repeat_weight) * scale(p) + self.repeat_weight * scale(own)


def top_k(scores: np.ndarray, k: int, exclude=None) -> np.ndarray:
    """Indices of the k highest scores per row; `exclude` is a sparse matrix of items to skip."""
    s = scores.astype(float, copy=True)
    if exclude is not None:
        s[exclude.nonzero()] = -np.inf
    idx = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(s, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


def default_candidates(lines: pd.DataFrame) -> list:
    """Settings chosen on a validation window of the demo store (see the notebook)."""
    return [Popularity(lines), BuyAgain(), ItemKNN(50, 10.0), EASE(300.0), Hybrid(EASE(300.0), 0.5)]
