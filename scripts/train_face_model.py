#!/usr/bin/env python
"""Phase 3: train and evaluate the four-member facial recognition model.

Evaluation uses StratifiedGroupKFold so all augmented versions of one source
photo remain in the same fold. Real impostor photos select the rejection
threshold applied to the maximum predicted member probability.

Run: python scripts/train_face_model.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

try:
    # Package import used by notebooks and tests from the repository root.
    from scripts.image_features import (
        DETECTOR_MODEL,
        OUTPUT_CSV,
        create_detector,
        feature_vector_from_path,
        impostor_image_paths,
    )
except ModuleNotFoundError:
    # Direct-script import used by `python scripts/train_face_model.py`.
    from image_features import (
        DETECTOR_MODEL,
        OUTPUT_CSV,
        create_detector,
        feature_vector_from_path,
        impostor_image_paths,
    )

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "face_model.joblib"
METRICS_PATH = ROOT / "results" / "metrics" / "facial_recognition.json"


def make_classifier(random_state: int = 42) -> RandomForestClassifier:
    """Create the classifier used consistently in evaluation and final fitting."""
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def load_features(path: Path = OUTPUT_CSV) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the feature table required by the model."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/image_features.py first")
    features = pd.read_csv(path)
    feature_columns = [column for column in features if column.startswith("hog_")]
    required = {"member", "source_image_id", "expression", "augmented"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("Feature CSV has no HOG columns")
    if features["member"].nunique() != 4:
        raise ValueError("The facial model requires exactly four member identities")
    if features.groupby("member")["source_image_id"].nunique().min() < 3:
        raise ValueError("Every member needs three source images for 3-fold evaluation")
    return features, feature_columns


def grouped_cross_validation(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run three leakage-safe folds and return out-of-fold predictions."""
    x = features[feature_columns]
    y = features["member"].astype(str).to_numpy()
    groups = features["source_image_id"].astype(str).to_numpy()
    classes = sorted(np.unique(y).tolist())
    class_to_index = {label: index for index, label in enumerate(classes)}

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_predictions = np.empty(len(features), dtype=object)
    oof_confidences = np.zeros(len(features), dtype=float)
    oof_probabilities = np.zeros((len(features), len(classes)), dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise AssertionError("A source image leaked across train and test")

        model = make_classifier(random_state=42 + fold)
        model.fit(x.iloc[train_index], y[train_index])
        predictions = model.predict(x.iloc[test_index])
        raw_probabilities = model.predict_proba(x.iloc[test_index])
        aligned = np.zeros((len(test_index), len(classes)), dtype=float)
        for model_column, label in enumerate(model.classes_):
            aligned[:, class_to_index[str(label)]] = raw_probabilities[:, model_column]

        oof_predictions[test_index] = predictions
        oof_probabilities[test_index] = aligned
        oof_confidences[test_index] = aligned.max(axis=1)
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y[test_index], predictions)),
                "f1_macro": float(
                    f1_score(y[test_index], predictions, average="macro")
                ),
                "log_loss": float(
                    log_loss(y[test_index], aligned, labels=classes)
                ),
                "train_source_images": len(train_groups),
                "test_source_images": len(test_groups),
            }
        )

    return fold_metrics, oof_predictions, oof_confidences, oof_probabilities, classes


def impostor_confidences(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    """Score the real non-member photos with the final known-member classifier."""
    scores: list[dict[str, object]] = []
    paths = impostor_image_paths()
    if not paths:
        raise ValueError("At least one image_data/impostor image is required")
    with create_detector(DETECTOR_MODEL) as detector:
        for path in paths:
            feature_row, detection_score, _ = feature_vector_from_path(path, detector)
            frame = pd.DataFrame([feature_row])[feature_columns]
            probabilities = model.predict_proba(frame)[0]
            best_index = int(np.argmax(probabilities))
            scores.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "predicted_member": str(model.classes_[best_index]),
                    "member_confidence": float(probabilities[best_index]),
                    "face_detection_score": detection_score,
                }
            )
    return scores


def choose_threshold(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    known_confidences: np.ndarray,
    impostors: list[dict[str, object]],
) -> tuple[float, dict[str, float]]:
    """Choose the threshold balancing known acceptance and impostor rejection."""
    impostor_values = np.array(
        [float(item["member_confidence"]) for item in impostors], dtype=float
    )
    values = np.unique(np.concatenate([known_confidences, impostor_values]))
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                values,
                (values[:-1] + values[1:]) / 2 if len(values) > 1 else values,
            ]
        )
    )

    best_threshold = 0.65
    best_rates = {"known_acceptance_rate": 0.0, "impostor_rejection_rate": 0.0}
    best_key = (-1.0, -1.0, -1.0, -1.0)
    correct = predictions.astype(str) == true_labels.astype(str)
    for threshold in candidates:
        known_rate = float(np.mean(correct & (known_confidences >= threshold)))
        impostor_rate = float(np.mean(impostor_values < threshold))
        balanced_rate = (known_rate + impostor_rate) / 2
        key = (
            balanced_rate,
            min(known_rate, impostor_rate),
            known_rate,
            -abs(float(threshold) - 0.65),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_rates = {
                "known_acceptance_rate": known_rate,
                "impostor_rejection_rate": impostor_rate,
                "balanced_authentication_rate": balanced_rate,
            }
    return best_threshold, best_rates


def train_and_evaluate(
    feature_path: Path = OUTPUT_CSV,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate, fit the final model, select a threshold, and save artifacts."""
    features, feature_columns = load_features(feature_path)
    y = features["member"].astype(str).to_numpy()
    (
        folds,
        oof_predictions,
        oof_confidence,
        _,
        classes,
    ) = grouped_cross_validation(features, feature_columns)

    final_model = make_classifier()
    final_model.fit(features[feature_columns], y)
    impostors = impostor_confidences(final_model, feature_columns)
    threshold, threshold_rates = choose_threshold(
        y, oof_predictions, oof_confidence, impostors
    )

    metric_names = ["accuracy", "f1_macro", "log_loss"]
    summary = {
        f"{metric}_{suffix}": float(
            getattr(np, operation)([fold[metric] for fold in folds])
        )
        for metric in metric_names
        for suffix, operation in (("mean", "mean"), ("std", "std"))
    }
    source_counts = (
        features[["member", "source_image_id"]]
        .drop_duplicates()
        .groupby("member")
        .size()
        .to_dict()
    )
    metrics: dict[str, object] = {
        "model": "RandomForestClassifier",
        "feature_extraction": "MediaPipe face crop + scikit-image HOG",
        "cv_method": "StratifiedGroupKFold(n_splits=3, groups=source_image_id)",
        "classes": classes,
        "source_images": int(features["source_image_id"].nunique()),
        "feature_rows": int(len(features)),
        "hog_feature_count": len(feature_columns),
        "source_images_per_member": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "folds": folds,
        **summary,
        "confusion_matrix": confusion_matrix(
            y, oof_predictions.astype(str), labels=classes
        ).tolist(),
        "known_oof_confidences": [float(value) for value in oof_confidence],
        "known_oof_predictions": [str(value) for value in oof_predictions],
        "known_oof_true_labels": [str(value) for value in y],
        "confidence_threshold": threshold,
        **threshold_rates,
        "impostor_results": impostors,
        "limitations": (
            "Only 12 member source images and 2 impostor images are available; "
            "report cross-validation variability and avoid production claims."
        ),
    }

    bundle: dict[str, object] = {
        "model": final_model,
        "feature_columns": feature_columns,
        "confidence_threshold": threshold,
        "face_size": list((128, 128)),
        "feature_extraction": metrics["feature_extraction"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle, metrics


def predict_face(
    image_path: Path,
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    """Return identity, confidence, and known/unknown decision for one image."""
    bundle = joblib.load(model_path)
    with create_detector(DETECTOR_MODEL) as detector:
        features, detection_score, _ = feature_vector_from_path(image_path, detector)
    frame = pd.DataFrame([features])[bundle["feature_columns"]]
    probabilities = bundle["model"].predict_proba(frame)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    threshold = float(bundle["confidence_threshold"])
    return {
        "identity": str(bundle["model"].classes_[best_index]),
        "confidence": confidence,
        "threshold": threshold,
        "accepted": confidence >= threshold,
        "face_detection_score": detection_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    _, metrics = train_and_evaluate(args.features, args.model, args.metrics)
    print("Facial recognition model trained with grouped 3-fold CV")
    print(
        "  accuracy : "
        f"{metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}"
    )
    print(
        "  macro F1 : "
        f"{metrics['f1_macro_mean']:.3f} ± {metrics['f1_macro_std']:.3f}"
    )
    print(
        "  log loss : "
        f"{metrics['log_loss_mean']:.3f} ± {metrics['log_loss_std']:.3f}"
    )
    print(f"  threshold: {metrics['confidence_threshold']:.3f}")
    print(
        "  known accepted / impostors rejected: "
        f"{metrics['known_acceptance_rate']:.1%} / "
        f"{metrics['impostor_rejection_rate']:.1%}"
    )
    print(f"  model   -> {args.model.relative_to(ROOT)}")
    print(f"  metrics -> {args.metrics.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
