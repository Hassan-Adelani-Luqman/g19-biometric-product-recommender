#!/usr/bin/env python
"""Phase 4 - train & evaluate the Voiceprint (speaker) model.

Reads data/processed/audio_features.csv, trains a speaker classifier, evaluates it,
and saves the model plus its metrics.

Split: with only ~8 clips, a random split leaks because an augmented copy is nearly
identical to its original. We train on the "Yes, approve" clips and test on the
"Confirm transaction" clips, so each clip and its augmentations stay on one side and
the model must recognise the person, not the words. If both phrases aren't usable we
fall back to leave-one-clip-out cross-validation.

Threshold: the model outputs a probability per member; if the top probability is below
THRESHOLD the voice is treated as unknown (Access Denied). Phase 6 also checks the
predicted speaker matches the face identity.

Run:
    python scripts/train_voice_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from audio_features import FEATURE_COLUMNS, META_COLUMNS  # noqa: E402

FEATURES_CSV = ROOT / "data" / "processed" / "audio_features.csv"
MODEL_OUT = ROOT / "models" / "voice_model.joblib"
METRICS_OUT = ROOT / "results" / "metrics" / "voiceprint.json"

THRESHOLD = 0.50     # min top-probability to accept a voice as a known member
RANDOM_STATE = 42


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    )


def phrase_held_out(df: pd.DataFrame):
    """Return (train_df, test_df), or (None, None) if both phrases aren't usable."""
    train = df[df["phrase"] == "yes_approve"]
    test = df[df["phrase"] == "confirm_transaction"]
    if train.empty or test.empty:
        return None, None
    if set(train["member"]) != set(test["member"]):
        return None, None
    return train, test


def evaluate_holdout(train, test, classes):
    X_tr, y_tr = train[FEATURE_COLUMNS].to_numpy(), train["member"].to_numpy()
    X_te, y_te = test[FEATURE_COLUMNS].to_numpy(), test["member"].to_numpy()

    model = new_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)

    return {
        "split": "phrase_held_out (train=yes_approve, test=confirm_transaction)",
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro",
                                         labels=classes, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_te, proba, labels=classes)), 4),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
    }


def evaluate_cv(df, classes):
    """Leave-one-clip-out CV fallback; reports mean +/- std accuracy."""
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["member"].to_numpy()
    groups = df["source_clip_id"].to_numpy()

    accs = []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        m = new_model().fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(X[te])))

    accs = np.array(accs) if accs else np.array([float("nan")])
    return {
        "method": "leave_one_clip_out_cv",
        "accuracy_mean": round(float(np.nanmean(accs)), 4),
        "accuracy_std": round(float(np.nanstd(accs)), 4),
        "n_folds": int(len(accs)),
    }


def main() -> None:
    if not FEATURES_CSV.exists():
        print(f"Missing {FEATURES_CSV.relative_to(ROOT)}. "
              f"Run scripts/audio_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    classes = sorted(df["member"].unique())
    print(f"Loaded {len(df)} rows | {len(classes)} member(s): {classes}")

    if len(classes) < 2:
        print("Only one member present - speaker classification needs >=2 members.")
        print("Add the other members' clips, then re-run. Saving a model anyway so "
              "the pipeline is exercised end-to-end.")

    # evaluate: phrase-held-out is the headline (text-independent); leave-one-clip-out
    # CV is reported alongside it so the tiny-n variance is visible, not hidden.
    train, test = phrase_held_out(df)
    if train is not None and len(classes) >= 2:
        metrics = evaluate_holdout(train, test, classes)
        metrics["cross_val_logo"] = evaluate_cv(df, classes)
    else:
        metrics = evaluate_cv(df, classes)
    print("Evaluation:", json.dumps(metrics, indent=2))

    # final model: fit on all data, save a self-describing bundle
    final = new_model().fit(df[FEATURE_COLUMNS].to_numpy(), df["member"].to_numpy())
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "feature_columns": FEATURE_COLUMNS,
            "classes": list(final.classes_),
            "threshold": THRESHOLD,
            "sr": 16_000,
        },
        MODEL_OUT,
    )
    print(f"Saved model -> {MODEL_OUT.relative_to(ROOT)}")

    # metrics shard (see PLAN.md 3.7: write the shard, not results/metrics.json)
    shard = {
        "model": "voiceprint",
        "algorithm": "RandomForestClassifier(n_estimators=300)",
        "n_source_clips": int(df["source_clip_id"].nunique()),
        "n_rows_total": int(len(df)),
        "classes": classes,
        "decision_threshold": THRESHOLD,
        **metrics,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(shard, indent=2))
    print(f"Wrote metrics shard -> {METRICS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
