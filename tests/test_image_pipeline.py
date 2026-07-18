"""Focused checks for the Phase 3 image and facial-recognition deliverables."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from scripts.image_features import (
    OUTPUT_CSV,
    augment_face,
    create_detector,
    detect_and_crop_face,
    extract_hog_features,
    feature_vector_from_path,
    impostor_image_paths,
    member_image_paths,
)
from scripts.train_face_model import METRICS_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]


class ImagePipelineTests(unittest.TestCase):
    def test_member_inventory_is_complete(self) -> None:
        paths = member_image_paths()
        counts = pd.Series([path.parent.name for path in paths]).value_counts()
        self.assertEqual(len(paths), 12)
        self.assertEqual(len(counts), 4)
        self.assertTrue((counts == 3).all())
        self.assertEqual(len(impostor_image_paths()), 2)

    def test_detection_augmentations_and_hog_shape(self) -> None:
        path = member_image_paths()[0]
        with create_detector() as detector, Image.open(path) as image:
            face, score = detect_and_crop_face(image, detector)
        variants = augment_face(face)
        descriptor = extract_hog_features(variants["original"])
        self.assertGreater(score, 0.40)
        self.assertEqual(
            set(variants),
            {"original", "rotated_10deg", "horizontal_flip", "brightness_115"},
        )
        self.assertEqual(descriptor.shape, (8100,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_feature_csv_has_leakage_safe_groups(self) -> None:
        features = pd.read_csv(OUTPUT_CSV)
        hog_columns = [column for column in features if column.startswith("hog_")]
        self.assertEqual(features.shape, (48, 8108))
        self.assertEqual(len(hog_columns), 8100)
        self.assertEqual(features["source_image_id"].nunique(), 12)
        self.assertTrue((features.groupby("source_image_id").size() == 4).all())
        self.assertTrue(
            (features.groupby("member")["source_image_id"].nunique() == 3).all()
        )
        self.assertFalse(features.isna().any().any())

    def test_metrics_and_saved_model_are_complete(self) -> None:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        bundle = joblib.load(MODEL_PATH)
        self.assertEqual(metrics["source_images"], 12)
        self.assertEqual(metrics["feature_rows"], 48)
        self.assertEqual(len(metrics["folds"]), 3)
        self.assertEqual(len(metrics["classes"]), 4)
        self.assertGreaterEqual(metrics["accuracy_mean"], 0.75)
        self.assertGreaterEqual(metrics["impostor_rejection_rate"], 0.5)
        self.assertEqual(len(bundle["feature_columns"]), 8100)
        self.assertAlmostEqual(
            float(bundle["confidence_threshold"]),
            float(metrics["confidence_threshold"]),
        )

    def test_final_model_accepts_members_and_rejects_impostors(self) -> None:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        columns = bundle["feature_columns"]
        threshold = float(bundle["confidence_threshold"])

        neutral_paths = [
            path for path in member_image_paths() if path.stem.lower() == "neutral"
        ]
        with create_detector() as detector:
            for path in neutral_paths:
                row, _, _ = feature_vector_from_path(path, detector)
                probabilities = model.predict_proba(pd.DataFrame([row])[columns])[0]
                best = int(np.argmax(probabilities))
                self.assertEqual(str(model.classes_[best]), path.parent.name)
                self.assertGreaterEqual(float(probabilities[best]), threshold)

            for path in impostor_image_paths():
                row, _, _ = feature_vector_from_path(path, detector)
                confidence = float(
                    model.predict_proba(pd.DataFrame([row])[columns])[0].max()
                )
                self.assertLess(confidence, threshold)


if __name__ == "__main__":
    unittest.main()
