"""Content-based "similar products" from catalogue text: works with zero sales history."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def product_text(products: pd.DataFrame) -> pd.Series:
    """Weight fields by repetition: title and category matter most, description least."""
    def col(name):
        return products[name].fillna("").astype(str)
    return (col("title") + " ") * 3 + (col("category") + " ") * 2 + (col("tags") + " ") * 2 \
        + col("brand") + " " + col("description").str[:500]


def similar_products(products: pd.DataFrame, per_product: int = 6, price_band: float | None = 2.5) -> pd.DataFrame:
    """Top similar products by TF-IDF cosine similarity (word 1-2-grams plus character 3-5-grams, which
    handles spelling variants such as 'kurta' / 'kurtas' and Roman Urdu product names)."""
    if len(products) < 2:
        return pd.DataFrame(columns=["product_id", "title", "similar_id", "similar_title", "similarity"])
    text = product_text(products)
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True,
                            stop_words="english").fit_transform(text)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000,
                            sublinear_tf=True).fit_transform(products["title"].fillna("").str.lower())
    sim = 0.7 * linear_kernel(words) + 0.3 * linear_kernel(chars)
    np.fill_diagonal(sim, -1)

    price = pd.to_numeric(products["price"], errors="coerce").to_numpy()
    if price_band and np.isfinite(price).sum() > len(price) * 0.5:
        # Prefer products in a similar price range (a 5,000 rupee suit is not a substitute for a 500 rupee one).
        ratio = np.maximum(price[:, None], price[None, :]) / np.maximum(np.minimum(price[:, None], price[None, :]), 1e-9)
        sim = np.where(np.isfinite(ratio) & (ratio > price_band), sim * 0.5, sim)

    k = min(per_product, len(products) - 1)
    top = np.argsort(-sim, axis=1)[:, :k]
    rows = []
    for i, neighbours in enumerate(top):
        for j in neighbours:
            if sim[i, j] <= 0.05:
                continue
            rows.append({"product_id": products["product_id"].iat[i], "title": products["title"].iat[i],
                         "similar_id": products["product_id"].iat[j], "similar_title": products["title"].iat[j],
                         "similarity": round(float(sim[i, j]), 3), "similar_url": products["url"].iat[j]})
    return pd.DataFrame(rows)
