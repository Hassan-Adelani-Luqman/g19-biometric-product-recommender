#!/usr/bin/env python
"""Phase 2 - tabular merge, cleaning, and the modelling table.

Produces data/processed/merged_dataset.csv: one row per transaction, enriched
with the customer's (aggregated) social profile. Target = product_category.

Design decisions (justified inline, mirrored in notebooks/01_eda_and_merge.ipynb):

  Join key   : customer_social_profiles.customer_id_new is "A" + digits and
               equals customer_transactions.customer_id_legacy once the "A" is
               stripped. Verified: every id matches ^A\\d+$.

  Grain      : social is one-to-many per customer (several platforms per person,
               plus 5 exact duplicate rows); transactions is one-to-many per
               customer (several purchases). A raw row-join would multiply rows,
               so social is FIRST aggregated to one row per customer, THEN joined.

  Join type  : LEFT (transactions is the base). The target lives in transactions,
               so we keep every transaction/label. Customers with no social row
               get imputed social features plus has_social_profile = False, which
               preserves all labels while flagging the imputation for the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "merged_dataset.csv"

SENTIMENTS = ["Negative", "Neutral", "Positive"]


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

    n_null = int(tx["customer_rating"].isna().sum())
    median_rating = tx["customer_rating"].median()
    tx["customer_rating"] = tx["customer_rating"].fillna(median_rating)
    print(f"  transactions: imputed {n_null} null customer_rating with median={median_rating}")

    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"])
    return tx


def merge(tx: pd.DataFrame, soc_agg: pd.DataFrame) -> pd.DataFrame:
    merged = tx.merge(soc_agg, on="customer_id", how="left", indicator=True)
    merged["has_social_profile"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")

    # Impute social features for transactions whose customer has no social row.
    merged["n_platforms"] = merged["n_platforms"].fillna(0).astype(int)
    for col in ["avg_engagement", "avg_purchase_interest"]:
        merged[col] = merged[col].fillna(round(soc_agg[col].mean(), 2))
    for col in ["primary_platform", "dominant_sentiment"]:
        merged[col] = merged[col].fillna("Unknown")

    # Final tidy dtypes.
    for col in ["product_category", "primary_platform", "dominant_sentiment"]:
        merged[col] = merged[col].astype("category")

    ordered = [
        "customer_id", "transaction_id", "purchase_date", "purchase_amount",
        "customer_rating", "avg_engagement", "avg_purchase_interest",
        "n_platforms", "primary_platform", "dominant_sentiment",
        "has_social_profile", "product_category",
    ]
    return merged[ordered]


def validate(tx: pd.DataFrame, merged: pd.DataFrame) -> None:
    print("\n  --- post-merge validation ---")
    print(f"  rows: transactions={len(tx)}  merged={len(merged)}  (equal => no row explosion)")
    assert len(merged) == len(tx), "row count changed: the join exploded"
    assert merged["transaction_id"].is_unique, "transaction_id no longer unique"

    nulls = merged.isna().sum()
    nulls = nulls[nulls > 0]
    print(f"  remaining nulls: {dict(nulls) if len(nulls) else 'none'}")
    assert nulls.empty, "unexpected nulls after cleaning"

    n_social = int(merged["has_social_profile"].sum())
    print(f"  transactions with real social profile: {n_social}/{len(merged)} "
          f"({100 * n_social / len(merged):.0f}%); imputed: {len(merged) - n_social}")

    # spot-check one customer: merged social features must match the aggregation
    cid = merged.loc[merged["has_social_profile"], "customer_id"].iloc[0]
    print(f"  spot-check customer {cid}:")
    print(merged[merged["customer_id"] == cid][
        ["transaction_id", "product_category", "avg_engagement", "primary_platform", "dominant_sentiment"]
    ].to_string(index=False))

    print("\n  target (product_category) distribution:")
    print(merged["product_category"].value_counts().to_string())


def main() -> None:
    print("Phase 2: merge + clean")
    soc, tx = load_raw()
    soc_agg = clean_social(soc)
    tx = clean_transactions(tx)
    merged = merge(tx, soc_agg)
    validate(tx, merged)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\n  saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
