#!/usr/bin/env python
"""Phase 3: detect faces, augment them, and extract HOG features.

MediaPipe finds and crops the face. The cropped face is resized to 128x128,
then scikit-image HOG converts it into numerical features for the classifier.

Run: python scripts/image_features.py
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "image_data"
AUGMENTED_ROOT = IMAGE_ROOT / "augmented"
OUTPUT_CSV = ROOT / "data" / "processed" / "image_features.csv"
DETECTOR_MODEL = ROOT / "models" / "blaze_face_short_range.tflite"
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXCLUDED_FOLDERS = {"augmented", "impostor"}
FACE_SIZE = (128, 128)
HOG_SETTINGS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def ensure_detector_model(model_path: Path = DETECTOR_MODEL) -> Path:
    """Download the official MediaPipe detector model when it is not cached."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face detector -> {model_path.relative_to(ROOT)}")
    urllib.request.urlretrieve(DETECTOR_URL, model_path)
    return model_path


def create_detector(model_path: Path = DETECTOR_MODEL) -> vision.FaceDetector:
    """Create the MediaPipe short-range face detector used by the pipeline."""
    model_path = ensure_detector_model(model_path)
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path.resolve())),
        min_detection_confidence=0.40,
    )
    return vision.FaceDetector.create_from_options(options)


def member_image_paths(image_root: Path = IMAGE_ROOT) -> list[Path]:
    """Return source images from member folders, excluding impostors/outputs."""
    paths: list[Path] = []
    for member_dir in sorted(image_root.iterdir()):
        if not member_dir.is_dir() or member_dir.name in EXCLUDED_FOLDERS:
            continue
        paths.extend(
            path
            for path in sorted(member_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return paths


def impostor_image_paths(image_root: Path = IMAGE_ROOT) -> list[Path]:
    """Return real non-member images reserved for threshold validation."""
    folder = image_root / "impostor"
    if not folder.exists():
        return []
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def detect_and_crop_face(
    image: Image.Image,
    detector: vision.FaceDetector,
    margin: float = 0.25,
) -> tuple[Image.Image, float]:
    """Detect the strongest face and return a square RGB crop plus confidence."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb_array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    result = detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
    )
    if not result.detections:
        raise ValueError("MediaPipe did not find a face")

    detection = max(result.detections, key=lambda item: item.categories[0].score)
    box = detection.bounding_box
    confidence = float(detection.categories[0].score)

    side = max(box.width, box.height)
    pad = int(round(side * margin))
    center_x = box.origin_x + box.width / 2
    center_y = box.origin_y + box.height / 2
    half_side = side / 2 + pad
    left = max(0, int(round(center_x - half_side)))
    top = max(0, int(round(center_y - half_side)))
    right = min(rgb.width, int(round(center_x + half_side)))
    bottom = min(rgb.height, int(round(center_y + half_side)))
    if right <= left or bottom <= top:
        raise ValueError("MediaPipe returned an invalid face box")

    crop = rgb.crop((left, top, right, bottom))
    return ImageOps.fit(crop, FACE_SIZE, method=Image.Resampling.LANCZOS), confidence


def augment_face(face: Image.Image) -> dict[str, Image.Image]:
    """Return the original face and three reproducible augmentations."""
    rgb = face.convert("RGB")
    return {
        "original": rgb,
        "rotated_10deg": rgb.rotate(
            10,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        ),
        "horizontal_flip": ImageOps.mirror(rgb),
        "brightness_115": ImageEnhance.Brightness(rgb).enhance(1.15),
    }


def extract_hog_features(face: Image.Image) -> np.ndarray:
    """Convert a face crop into a fixed-length HOG descriptor."""
    prepared = ImageOps.fit(
        face.convert("L"), FACE_SIZE, method=Image.Resampling.LANCZOS
    )
    gray = np.asarray(prepared, dtype=np.float32) / 255.0
    return hog(gray, feature_vector=True, **HOG_SETTINGS).astype(np.float32)


def feature_dict(face: Image.Image) -> dict[str, float]:
    """Return named HOG columns for one face image."""
    descriptor = extract_hog_features(face)
    return {
        f"hog_{index:04d}": float(value)
        for index, value in enumerate(descriptor)
    }


def feature_vector_from_path(
    image_path: Path,
    detector: vision.FaceDetector,
) -> tuple[dict[str, float], float, Image.Image]:
    """Extract inference-ready HOG features from one image path."""
    with Image.open(image_path) as image:
        face, detection_score = detect_and_crop_face(image, detector)
    return feature_dict(face), detection_score, face


def process_image_dataset(
    image_root: Path = IMAGE_ROOT,
    output_csv: Path = OUTPUT_CSV,
    augmented_root: Path = AUGMENTED_ROOT,
) -> pd.DataFrame:
    """Process all member images and save one row per original/augmentation."""
    source_paths = member_image_paths(image_root)
    if not source_paths:
        raise ValueError(f"No member images found under {image_root}")

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    augmented_root.mkdir(parents=True, exist_ok=True)

    with create_detector() as detector:
        for source_path in source_paths:
            member = source_path.parent.name
            expression = source_path.stem.lower().replace("suprised", "surprised")
            source_image_id = f"{member}/{source_path.stem}"
            try:
                features, detection_score, face = feature_vector_from_path(
                    source_path, detector
                )
            except Exception as error:  # noqa: BLE001 - report every bad source together
                failures.append(f"{source_path.relative_to(ROOT)}: {error}")
                continue

            variants = augment_face(face)
            for variant, variant_image in variants.items():
                member_output = augmented_root / member
                member_output.mkdir(parents=True, exist_ok=True)
                processed_path = member_output / f"{source_path.stem}_aug_{variant}.jpg"
                variant_image.save(processed_path, "JPEG", quality=92)
                variant_features = (
                    features if variant == "original" else feature_dict(variant_image)
                )
                rows.append(
                    {
                        "member": member,
                        "expression": expression,
                        "variant": variant,
                        "augmented": variant != "original",
                        "source_image_id": source_image_id,
                        "source_file": str(source_path.relative_to(ROOT)),
                        "processed_file": str(processed_path.relative_to(ROOT)),
                        "face_detection_score": detection_score,
                        **variant_features,
                    }
                )

    if failures:
        raise RuntimeError("Face extraction failed:\n- " + "\n- ".join(failures))

    features = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--augmented-root", type=Path, default=AUGMENTED_ROOT)
    args = parser.parse_args()

    features = process_image_dataset(
        image_root=args.image_root,
        output_csv=args.output,
        augmented_root=args.augmented_root,
    )
    source_count = features["source_image_id"].nunique()
    feature_count = sum(column.startswith("hog_") for column in features.columns)
    print(
        f"Saved {len(features)} rows from {source_count} source images "
        f"with {feature_count} HOG features -> {args.output.relative_to(ROOT)}"
    )
    print(features.groupby(["member", "variant"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
