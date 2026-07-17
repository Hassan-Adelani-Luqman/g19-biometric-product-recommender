#!/usr/bin/env python
r"""Phase 2 - tabular merge, cleaning, and the modelling table.

Produces data/processed/merged_dataset.csv: one row per transaction (for customers who
have a social profile), enriched with that customer's aggregated social features.

PREDICTION FRAMING (this drives every design choice).
The model predicts the product a customer would buy "when visiting your social media
sites" -- i.e. at social-visit time, BEFORE any purchase happens. So only the SOCIAL
features are legitimately available as predictors; the transaction columns are
post-purchase and must NOT feed the model (they would leak the outcome). Column roles:

    IDENTIFIERS     : customer_id, transaction_id
    PREDICTORS      : avg_engagement, avg_purchase_interest, n_platforms,
                      primary_platform, dominant_sentiment            (from social)
    TARGET          : product_category                                (from transactions)
    EXCLUDED (leak) : purchase_date, purchase_amount, customer_rating (post-purchase)

All columns are kept in the CSV for reference/EDA, but Phase 5 must train on PREDICTORS
only. The roles are exported as constants below so Phase 5 can import them directly.

MERGE DESIGN.
  Join key : customer_id_new ("A"+digits) == customer_id_legacy once "A" is stripped
             (verified: every id matches ^A\d+$).
  Grain    : social is one-to-many per customer (several platforms, plus 5 exact
             duplicate rows and repeated (customer, platform) readings); transactions is
             one-to-many per customer. Social is first AGGREGATED to one row per customer
             -- only EXACT duplicate rows are dropped; repeated (customer, platform)
             readings are summarised (mean / mode), not discarded -- then joined.
  Join type: INNER. The task predicts FROM the social profile, so the modelling
             population is exactly the customers who have a social profile AND purchased.
             Inner join (a) matches that framing and (b) avoids fabricating the very
             predictor features by imputation, which a LEFT join would force on the 14
             customers (33 transactions) that have no social profile at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "merged_dataset.csv"

# --- column roles (import these in Phase 5) ---
IDENTIFIERS = ["customer_id", "transaction_id"]
PREDICTORS = [
    "avg_engagement", "avg_purchase_interest", "n_platforms",
    "primary_platform", "dominant_sentiment",
]
TARGET = "product_category"
EXCLUDED_LEAKAGE = ["purchase_date", "purchase_amount", "customer_rating"]

COLUMN_ORDER = IDENTIFIERS + PREDICTORS + EXCLUDED_LEAKAGE + [TARGET]


def _mode(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    soc = pd.read_csv(RAW / "customer_social_profiles.csv")
    tx = pd.read_csv(RAW / "customer_transactions.csv")
    return soc, tx


def clean_social(soc: pd.DataFrame) -> pd.DataFrame:
    """Dedup, derive the numeric customer id, aggregate to one row per customer."""
    n0 = len(soc)
    soc = soc.drop_duplicates().copy()
    print(f"  social: dropped {n0 - len(soc)} exact duplicate rows -> {len(soc)} rows")

    assert soc["customer_id_new"].str.fullmatch(r"A\d+").all(), "unexpected id format"
    soc["customer_id"] = soc["customer_id_new"].str[1:].astype(int)

    # Repeated (customer, platform) readings are kept and summarised, not treated as errors.
    repeats = int(soc.duplicated(["customer_id", "social_media_platform"]).sum())
    print(f"  social: {repeats} repeated (customer, platform) readings -> summarised, not dropped")

    agg = (
        soc.groupby("customer_id")
        .agg(
            avg_engagement=("engagement_score", "mean"),
            avg_purchase_interest=("purchase_interest_score", "mean"),
            n_platforms=("social_media_platform", "nunique"),
            primary_platform=("social_media_platform", _mode),
            dominant_sentiment=("review_sentiment", _mode),
        )
        .reset_index()
    )
    agg["avg_engagement"] = agg["avg_engagement"].round(2)
    agg["avg_purchase_interest"] = agg["avg_purchase_interest"].round(2)
    print(f"  social: aggregated to {len(agg)} unique customers")
    return agg


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    print(f"  transactions: {tx.duplicated().sum()} exact duplicate rows")
    tx["customer_id"] = tx["customer_id_legacy"].astype(int)

    # customer_rating is an EXCLUDED (post-purchase) column, but we still impute its 10
    # nulls so the merged dataset is complete for EDA and the null-handling checks.
    n_null = int(tx["customer_rating"].isna().sum())
    median_rating = tx["customer_rating"].median()
    tx["customer_rating"] = tx["customer_rating"].fillna(median_rating)
    print(f"  transactions: imputed {n_null} null customer_rating with median={median_rating} (excluded feature)")

    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"])
    return tx


def merge(tx: pd.DataFrame, soc_agg: pd.DataFrame) -> pd.DataFrame:
    """INNER join: keep only transactions whose customer has a social profile."""
    merged = tx.merge(soc_agg, on="customer_id", how="inner")
    for col in ["product_category", "primary_platform", "dominant_sentiment"]:
        merged[col] = merged[col].astype("category")
    return merged[COLUMN_ORDER]


def validate(tx: pd.DataFrame, soc_agg: pd.DataFrame, merged: pd.DataFrame) -> None:
    print("\n  --- post-merge validation ---")
    expected = int(tx["customer_id"].isin(set(soc_agg["customer_id"])).sum())
    dropped = len(tx) - len(merged)
    print(f"  rows: transactions={len(tx)} -> merged={len(merged)}  "
          f"(inner join dropped {dropped} transactions from customers with no social profile)")
    assert len(merged) == expected, "row count != number of transactions with a social profile"
    assert merged["transaction_id"].is_unique, "transaction_id no longer unique"
    assert merged["primary_platform"].notna().all(), "a social feature is missing"

    nulls = merged.isna().sum()
    nulls = nulls[nulls > 0]
    print(f"  remaining nulls: {dict(nulls) if len(nulls) else 'none'}")
    assert nulls.empty, "unexpected nulls after cleaning"

    # spot-check one customer: merged social features must match the aggregation
    cid = merged["customer_id"].iloc[0]
    src = soc_agg[soc_agg["customer_id"] == cid][["avg_engagement", "primary_platform"]].iloc[0]
    got = merged[merged["customer_id"] == cid][["avg_engagement", "primary_platform"]].iloc[0]
    print(f"  spot-check customer {cid}: agg={src.to_dict()} == merged={got.to_dict()} "
          f"-> {bool((src.values == got.values).all())}")

    print("  column roles:")
    print(f"    predictors : {PREDICTORS}")
    print(f"    target     : {TARGET}")
    print(f"    excluded   : {EXCLUDED_LEAKAGE}  (post-purchase - not model inputs)")
    print("\n  target (product_category) distribution:")
    print(merged["product_category"].value_counts().to_string())


def main() -> None:
    print("Phase 2: merge + clean")
    soc, tx = load_raw()
    soc_agg = clean_social(soc)
    tx = clean_transactions(tx)
    merged = merge(tx, soc_agg)
    validate(tx, soc_agg, merged)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\n  saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
