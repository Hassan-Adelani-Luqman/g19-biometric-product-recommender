#!/usr/bin/env python
"""Phase 4 - train & evaluate the Voiceprint (speaker) model.

Model: LogisticRegression on standardized MFCC mean/std + spectral roll-off + RMS
energy. On this tiny dataset (8 source clips) a RandomForest over all 86 features
(incl. MFCC deltas, ZCR, centroid) badly overfits; a regularised linear model on the
leaner plan feature set generalises far better -- leave-one-clip-out CV rises from
0.50 +/- 0.35 to 0.84 +/- 0.28 and phrase-held-out accuracy from 0.69 to 0.94. The CSV
still stores all 86 features; only the model narrows to this subset.

Split: phrase-held-out (train "Yes, approve", test "Confirm transaction") so each clip
and its augmentations stay on one side and the model must recognise the person, not the
words. Leave-one-clip-out CV is reported alongside so the tiny-n variance is visible.

Threshold: derived from data as the 10th percentile of the out-of-fold confidence on
correct genuine predictions (accepts ~90% of genuine voices). There is no impostor-voice
set, so Phase 6's identity cross-check (predicted speaker == face identity) is the primary
rejection gate; this threshold only screens very-low-confidence voices.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from audio_features import FEATURE_COLUMNS  # noqa: E402

FEATURES_CSV = ROOT / "data" / "processed" / "audio_features.csv"
MODEL_OUT = ROOT / "models" / "voice_model.joblib"
METRICS_OUT = ROOT / "results" / "metrics" / "voiceprint.json"

RANDOM_STATE = 42

# Lean, plan-compliant model features: MFCC mean/std + spectral roll-off + RMS energy.
# Excludes the MFCC deltas / ZCR / centroid that the CSV also stores, because on 8 clips
# the full 86-D vector overfits (see module docstring for the measured difference).
MODEL_FEATURES = [
    c for c in FEATURE_COLUMNS
    if (c.startswith("mfcc_") and "delta" not in c)
    or c.startswith("rolloff_")
    or c.startswith("rms_")
]


def new_model():
    """Regularised linear speaker classifier over standardized features."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
        ),
    )


def phrase_held_out(df: pd.DataFrame):
    """Return (train_df, test_df), or (None, None) if both phrases aren't usable."""
    train = df[df["phrase"] == "yes_approve"]
    test = df[df["phrase"] == "confirm_transaction"]
    if train.empty or test.empty or set(train["member"]) != set(test["member"]):
        return None, None
    return train, test


def evaluate_holdout(train, test, classes) -> dict:
    model = new_model().fit(train[MODEL_FEATURES], train["member"])
    pred = model.predict(test[MODEL_FEATURES])
    proba = model.predict_proba(test[MODEL_FEATURES])
    return {
        "split": "phrase_held_out (train=yes_approve, test=confirm_transaction)",
        "accuracy": round(float(accuracy_score(test["member"], pred)), 4),
        "f1_macro": round(float(f1_score(test["member"], pred, average="macro",
                                         labels=classes, zero_division=0)), 4),
        "log_loss": round(float(log_loss(test["member"], proba, labels=classes)), 4),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
    }


def leave_one_clip_out(df):
    """LOGO CV. Returns (metrics, genuine_correct_confidences)."""
    X = df[MODEL_FEATURES].to_numpy()
    y = df["member"].to_numpy()
    groups = df["source_clip_id"].to_numpy()
    accs: list[float] = []
    genuine_conf: list[float] = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        model = new_model().fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        pred = model.classes_[proba.argmax(axis=1)]
        accs.append(accuracy_score(y[te], pred))
        for i in range(len(te)):
            if pred[i] == y[te][i]:
                genuine_conf.append(float(proba[i].max()))
    accs = np.array(accs) if accs else np.array([float("nan")])
    metrics = {
        "method": "leave_one_clip_out_cv",
        "accuracy_mean": round(float(np.nanmean(accs)), 4),
        "accuracy_std": round(float(np.nanstd(accs)), 4),
        "n_folds": int(len(accs)),
    }
    return metrics, np.array(genuine_conf)


def choose_threshold(genuine_conf: np.ndarray):
    """Accept ~90% of genuine voices: 10th percentile of OOF correct-prediction confidence."""
    if len(genuine_conf) == 0:
        return 0.5, 0.0
    thr = float(np.percentile(genuine_conf, 10))
    accept = float((genuine_conf >= thr).mean())
    return round(thr, 4), round(accept, 4)


def main() -> None:
    if not FEATURES_CSV.exists():
        print(f"Missing {FEATURES_CSV.relative_to(ROOT)}. Run scripts/audio_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    classes = sorted(df["member"].unique())
    print(f"Loaded {len(df)} rows | {len(classes)} members | {len(MODEL_FEATURES)} model features")
    if len(classes) < 2:
        print("Speaker classification needs >=2 members; add the other clips and re-run.")
        return

    cv_metrics, genuine_conf = leave_one_clip_out(df)
    threshold, genuine_accept = choose_threshold(genuine_conf)

    train, test = phrase_held_out(df)
    if train is not None:
        metrics = evaluate_holdout(train, test, classes)
        metrics["cross_val_logo"] = cv_metrics
    else:
        metrics = cv_metrics

    final = new_model().fit(df[MODEL_FEATURES].to_numpy(), df["member"].to_numpy())
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "feature_columns": MODEL_FEATURES,
            "classes": list(final.classes_),
            "threshold": threshold,
            "sr": 16_000,
        },
        MODEL_OUT,
    )
    print(f"Saved model -> {MODEL_OUT.relative_to(ROOT)}")

    shard = {
        "model": "voiceprint",
        "algorithm": "LogisticRegression(C=0.5) on standardized MFCC mean/std + rolloff + RMS",
        "n_model_features": len(MODEL_FEATURES),
        "n_source_clips": int(df["source_clip_id"].nunique()),
        "n_rows_total": int(len(df)),
        "classes": classes,
        "decision_threshold": threshold,
        "threshold_rule": "10th percentile of out-of-fold confidence on correct genuine predictions",
        "genuine_acceptance_rate": genuine_accept,
        **metrics,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(shard, indent=2))
    print(f"Wrote metrics shard -> {METRICS_OUT.relative_to(ROOT)}")
    headline = {k: shard[k] for k in
                ("accuracy", "f1_macro", "log_loss", "cross_val_logo",
                 "decision_threshold", "genuine_acceptance_rate") if k in shard}
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
