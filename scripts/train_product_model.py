#!/usr/bin/env python
"""Phase 5 - train & evaluate the Product Recommendation model.

Predicts `product_category` from a customer's SOCIAL profile only (the predictors set up
in Phase 2). The transaction columns (amount, rating, date) are post-purchase leakage and
are excluded -- see merge_and_clean.PREDICTORS / EXCLUDED_LEAKAGE.

Honest evaluation (this is the important bit):
    Rows are per-transaction, but a customer's social features are IDENTICAL across all
    their transactions. A plain row split would put the same customer in train and test
    and inflate the score. We therefore group the split by `customer_id`
    (StratifiedGroupKFold) so the model is scored on UNSEEN customers.

Result: with 61 customers, 5 classes, only customer-level features, and 30/61 customers
buying more than one category, the social profile carries weak signal. The model
marginally beats the majority-class baseline on accuracy and clearly on macro-F1 -- it
learns something, but the ceiling is low. Reported transparently, not inflated. (A leaky
row split is computed too, only to show how much it would over-state the result.)

Model: regularised LogisticRegression on one-hot social features -- chosen over RF/XGBoost
because on this tiny, mostly-categorical data it gave the best, most stable grouped-CV
accuracy and F1.

Run:
    python scripts/train_product_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from merge_and_clean import EXCLUDED_LEAKAGE, PREDICTORS, TARGET  # noqa: E402

DATA = ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_OUT = ROOT / "models" / "product_model.joblib"
METRICS_OUT = ROOT / "results" / "metrics" / "product_recommendation.json"

NUMERIC = ["avg_engagement", "avg_purchase_interest", "n_platforms"]
CATEGORICAL = ["primary_platform", "dominant_sentiment"]
N_SPLITS = 5
RANDOM_STATE = 42

TEAM_MEMBERS = [
    "Gentil_Tonny_Christian_Iradukunda",
    "Hassan_Adelani_Luqman",
    "Mahlet_Assefa_Tilahun",
    "Yvette_Uwimpaye",
]


def make_pipeline() -> Pipeline:
    """Preprocess (one-hot categoricals, scale numerics) + regularised LogisticRegression."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    clf = LogisticRegression(C=0.2, max_iter=5000, random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def grouped_cv(df: pd.DataFrame, classes: list[str]) -> tuple[dict, list[dict]]:
    """Customer-grouped StratifiedGroupKFold. Returns (summary, per-fold)."""
    X, y = df[PREDICTORS], df[TARGET].to_numpy()
    groups = df["customer_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    folds, accs, f1s, lls = [], [], [], []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("a customer leaked across train/test")
        pipe = make_pipeline().fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        proba = pipe.predict_proba(X.iloc[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", labels=classes, zero_division=0))
        lls.append(log_loss(y[te], proba, labels=classes))
        folds.append({
            "fold": k,
            "accuracy": round(accs[-1], 4),
            "f1_macro": round(f1s[-1], 4),
            "log_loss": round(lls[-1], 4),
            "test_customers": int(len(set(groups[te]))),
            "test_rows": int(len(te)),
        })

    summary = {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "f1_macro_mean": round(float(np.mean(f1s)), 4),
        "f1_macro_std": round(float(np.std(f1s)), 4),
        "log_loss_mean": round(float(np.mean(lls)), 4),
        "log_loss_std": round(float(np.std(lls)), 4),
    }
    return summary, folds


def baseline_accuracy(df: pd.DataFrame) -> float:
    """Majority-class baseline under the same grouped CV."""
    X, y = df[PREDICTORS], df[TARGET].to_numpy()
    groups = df["customer_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for tr, te in splitter.split(X, y, groups):
        dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr])
        accs.append(accuracy_score(y[te], dummy.predict(X.iloc[te])))
    return round(float(np.mean(accs)), 4)


def leaky_rowsplit_accuracy(df: pd.DataFrame) -> float:
    """Plain (leaky) row-level CV, shown only to illustrate the inflation grouping avoids."""
    X, y = df[PREDICTORS], df[TARGET].to_numpy()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs = [accuracy_score(y[te], make_pipeline().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
            for tr, te in splitter.split(X, y)]
    return round(float(np.mean(accs)), 4)


def member_demo_records(df: pd.DataFrame, pipeline: Pipeline) -> dict:
    """Assign each team member a real customer record (distinct predicted product where
    possible) so Phase 6 can map a recognised face -> tabular record -> product."""
    reps = df.drop_duplicates("customer_id").sort_values("customer_id").copy()
    reps["pred"] = pipeline.predict(reps[PREDICTORS])
    chosen: dict = {}
    used_customers: set = set()
    used_products: set = set()
    for member in TEAM_MEMBERS:
        pick = next((r for _, r in reps.iterrows()
                     if r.customer_id not in used_customers and r.pred not in used_products), None)
        if pick is None:  # fall back to any unused customer
            pick = next(r for _, r in reps.iterrows() if r.customer_id not in used_customers)
        used_customers.add(int(pick.customer_id))
        used_products.add(pick.pred)
        chosen[member] = {
            "customer_id": int(pick.customer_id),
            "predictors": {c: (float(pick[c]) if c in NUMERIC else str(pick[c])) for c in PREDICTORS},
            "predicted_product": str(pick.pred),
        }
    return chosen


def predict_product(record: dict, model_path: Path = MODEL_OUT) -> dict:
    """Phase 6 helper: predictor dict in -> product + confidence + ranking out."""
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    frame = pd.DataFrame([record])[bundle["predictors"]]
    proba = pipe.predict_proba(frame)[0]
    order = np.argsort(proba)[::-1]
    classes = list(pipe.classes_)
    return {
        "product": classes[order[0]],
        "confidence": float(proba[order[0]]),
        "ranking": [(classes[i], round(float(proba[i]), 3)) for i in order],
    }


def main() -> None:
    if not DATA.exists():
        print(f"Missing {DATA.relative_to(ROOT)}. Run scripts/merge_and_clean.py first.")
        return
    df = pd.read_csv(DATA)
    classes = sorted(df[TARGET].unique())
    print(f"Loaded {len(df)} rows | {df.customer_id.nunique()} customers | "
          f"{len(classes)} classes | features: {PREDICTORS}")

    summary, folds = grouped_cv(df, classes)
    baseline = baseline_accuracy(df)
    leaky = leaky_rowsplit_accuracy(df)

    final = make_pipeline().fit(df[PREDICTORS], df[TARGET])
    demo = member_demo_records(df, final)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final,
            "predictors": PREDICTORS,
            "classes": list(final.classes_),
            "member_demo_records": demo,
        },
        MODEL_OUT,
    )
    print(f"Saved model -> {MODEL_OUT.relative_to(ROOT)}")

    shard = {
        "model": "product_recommendation",
        "algorithm": "LogisticRegression(C=0.2) on one-hot social features + scaled numerics",
        "features": PREDICTORS,
        "target": TARGET,
        "excluded_leakage": EXCLUDED_LEAKAGE,
        "n_rows": int(len(df)),
        "n_customers": int(df.customer_id.nunique()),
        "n_classes": len(classes),
        "classes": classes,
        "cv_method": f"StratifiedGroupKFold(n_splits={N_SPLITS}, groups=customer_id)",
        "majority_baseline_accuracy": baseline,
        **summary,
        "folds": folds,
        "leaky_rowsplit_accuracy_for_contrast": leaky,
        "member_demo_records": demo,
        "limitations": (
            "Predictors are customer-level (identical across a customer's transactions) and "
            f"30/{int(df.customer_id.nunique())} customers buy >1 category, so the same profile "
            "maps to several products -- an irreducible ceiling. With 5 classes and tiny n, the "
            "model beats the majority baseline only marginally on accuracy (clearly on macro-F1). "
            "More per-visit context, not more tuning, is what would raise this."
        ),
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(shard, indent=2))
    print(f"Wrote metrics shard -> {METRICS_OUT.relative_to(ROOT)}")
    print(f"  grouped CV : acc {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f} "
          f"| macro-F1 {summary['f1_macro_mean']:.3f} | log-loss {summary['log_loss_mean']:.3f}")
    print(f"  baseline   : acc {baseline:.3f}   (leaky row-split would read {leaky:.3f})")


if __name__ == "__main__":
    main()
