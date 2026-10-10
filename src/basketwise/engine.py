"""End-to-end run on one store's orders: clean, validate on its own history, pick a model, produce outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import cleaning, evaluation, models, outputs

MIN_CUSTOMERS_FOR_PERSONAL = 50   # below this, personal picks are not reliable enough to sell
MIN_EVAL_CUSTOMERS = 30           # customers needed in the test window to report a score


@dataclass
class StoreResults:
    summary: dict
    cleaning: cleaning.CleaningReport
    evaluation: pd.DataFrame | None
    proof: list[str]
    chosen: dict[str, str]
    names: pd.Series
    data: models.Interactions | None
    model_everything: object | None
    model_discovery: object | None
    values: pd.DataFrame
    fbt: pd.DataFrame
    bundles: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _best(table: pd.DataFrame, mode: str) -> str:
    sub = table[table["mode"] == mode]
    return sub.sort_values(["ndcg", "hit_rate"], ascending=False)["model"].iloc[0]


def run(lines: pd.DataFrame, evaluate: bool = True) -> StoreResults:
    clean, report = cleaning.clean(lines)
    summary = cleaning.summary(clean)
    names = clean.drop_duplicates("product_id").set_index("product_id")["product_name"]
    values = outputs.customer_value(clean)
    fbt = outputs.frequently_bought_together(clean, names, min_orders=max(3, min(10, summary["orders"] // 500)))
    warnings, proof, table = [], [], None
    chosen = {"Everything": "EASE + buy again", "Discovery": "EASE"}

    if summary["customers"] < MIN_CUSTOMERS_FOR_PERSONAL:
        warnings.append(f"Only {summary['customers']} identified customers: personal picks need at least "
                        f"{MIN_CUSTOMERS_FOR_PERSONAL}. Frequently-bought-together and bundles still work.")
        return StoreResults(summary, report, None, proof, chosen, names, None, None, None, values, fbt,
                            outputs.bundles(fbt, clean), warnings)

    if evaluate:
        split = evaluation.temporal_split(clean)
        if len(split.test_rows) >= MIN_EVAL_CUSTOMERS:
            table = evaluation.evaluate(split, models.default_candidates(split.train_lines))
            chosen = {mode: _best(table, mode) for mode in ["Everything", "Discovery"]}
            best = table[(table["mode"] == "Discovery") & (table["model"] == chosen["Discovery"])].iloc[0]
            pop = table[(table["mode"] == "Discovery") & (table["model"] == "Popular now")].iloc[0]
            proof.append(
                f"We hid your last {split.test_days} days of orders and asked the engine to predict them. For "
                f"{best['hit_rate']:.0%} of your {int(best['customers']):,} returning customers, at least one of the "
                f"10 new products we suggested was something they really went on to buy.")
            if pop["hit_rate"] > 0:
                proof.append(f"That is {best['hit_rate'] / pop['hit_rate']:.1f}× better than showing everyone "
                             f"your best-sellers ({pop['hit_rate']:.0%}).")
        else:
            warnings.append("Not enough recent repeat customers to measure accuracy on your data; "
                            "recommendations use settings validated on a demo store.")

    data = models.build_interactions(clean)
    lookup = {m.name: m for m in models.default_candidates(clean)}
    model_everything = lookup[chosen["Everything"]].fit(data)
    model_discovery = lookup[chosen["Discovery"]].fit(data) if chosen["Discovery"] != chosen["Everything"] \
        else model_everything
    return StoreResults(summary, report, table, proof, chosen, names, data, model_everything, model_discovery,
                        values, fbt, outputs.bundles(fbt, clean), warnings)
