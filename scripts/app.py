#!/usr/bin/env python
"""Phase 6 - Multimodal authentication + product recommendation CLI.

Replicates the assignment flowchart exactly:

    face image --> Facial Recognition --fail--> ACCESS DENIED
                                       --pass--> Product Recommendation (held)
    voice clip --> Voice Validation ----fail--> ACCESS DENIED
                   (low confidence OR predicted speaker != recognised face)
                                    ----pass--> display the predicted product

The identity cross-check is the point of the multimodal design: an approved voice only
unlocks the transaction if it belongs to the SAME member the face was recognised as.

Usage:
    python scripts/app.py --face <image> --voice <audio.wav>
    python scripts/app.py                      # prompts for the two paths

Exit codes: 0 = access granted, 1 = access denied, 3 = bad input file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from train_face_model import predict_face             # noqa: E402
from audio_features import extract_features_for_file  # noqa: E402

FACE_MODEL = ROOT / "models" / "face_model.joblib"
VOICE_MODEL = ROOT / "models" / "voice_model.joblib"
PRODUCT_MODEL = ROOT / "models" / "product_model.joblib"


def voice_identity(audio_path: Path, bundle: dict) -> tuple[str, float]:
    """Return (predicted speaker, confidence) for a voice clip."""
    feats = extract_features_for_file(audio_path)
    # select columns in the trained order, then pass a bare array: the voice model's
    # scaler was fitted on a numpy array, so a named DataFrame would warn.
    frame = pd.DataFrame([feats])[bundle["feature_columns"]].to_numpy()
    proba = bundle["model"].predict_proba(frame)[0]
    idx = int(np.argmax(proba))
    return str(bundle["model"].classes_[idx]), float(proba[idx])


def product_for(identity: str, bundle: dict) -> dict | None:
    """Look up the member's demo customer record and predict their product."""
    record = bundle["member_demo_records"].get(identity)
    if record is None:
        return None
    frame = pd.DataFrame([record["predictors"]])[bundle["predictors"]]
    proba = bundle["pipeline"].predict_proba(frame)[0]
    idx = int(np.argmax(proba))
    return {
        "customer_id": record["customer_id"],
        "product": str(bundle["pipeline"].classes_[idx]),
        "confidence": float(proba[idx]),
    }


def deny(reason: str) -> int:
    print(f"\n>>> ACCESS DENIED - {reason}")
    return 1


def run(face_path: Path, voice_path: Path) -> int:
    print("=" * 58)
    print("  Multimodal Auth & Product Recommendation")
    print("=" * 58)

    # --- Step 1: Facial Recognition ---
    print(f"\n[1] Facial Recognition   (image: {face_path.name})")
    try:
        face = predict_face(face_path, FACE_MODEL)
    except Exception as error:
        print(f"    could not process image: {error}")
        return deny("face image unreadable / no face detected")
    print(f"    identity   : {face['identity']}")
    print(f"    confidence : {face['confidence']:.2f}  (threshold {face['threshold']:.2f})")
    if not face["accepted"]:
        return deny("face not recognised (confidence below threshold)")
    identity = face["identity"]
    print("    -> PASS")

    # --- Step 2: Product Recommendation (held until voice passes) ---
    print("\n[2] Product Recommendation   (held)")
    product_bundle = joblib.load(PRODUCT_MODEL)
    product = product_for(identity, product_bundle)
    if product is None:
        return deny(f"no customer record mapped for {identity}")
    print(f"    {identity} -> customer {product['customer_id']} -> "
          f"'{product['product']}'  (not shown yet)")

    # --- Step 3: Voice Validation + identity cross-check ---
    print(f"\n[3] Voice Validation   (audio: {voice_path.name})")
    voice_bundle = joblib.load(VOICE_MODEL)
    try:
        speaker, vconf = voice_identity(voice_path, voice_bundle)
    except Exception as error:
        print(f"    could not process audio: {error}")
        return deny("audio unreadable")
    vthr = float(voice_bundle["threshold"])
    print(f"    speaker    : {speaker}")
    print(f"    confidence : {vconf:.2f}  (threshold {vthr:.2f})")
    if vconf < vthr:
        return deny("voice confidence below threshold")
    if speaker != identity:
        print(f"    identity   : voice '{speaker}' != face '{identity}'")
        return deny("voice does not match the recognised face")
    print(f"    identity   : voice == face ({identity})  OK")
    print("    -> PASS")

    # --- Access granted ---
    print("\n" + "=" * 58)
    print("  ACCESS GRANTED")
    print("=" * 58)
    print(f"  Member            : {identity}")
    print(f"  Predicted product : {product['product']}  "
          f"(model confidence {product['confidence']:.2f})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal auth + product recommendation")
    parser.add_argument("--face", type=Path, help="path to a face image")
    parser.add_argument("--voice", type=Path, help="path to a voice .wav")
    args = parser.parse_args()

    face_path = args.face or Path(input("Face image path: ").strip())
    voice_path = args.voice or Path(input("Voice audio path: ").strip())
    if not face_path.exists():
        print(f"face image not found: {face_path}")
        sys.exit(3)
    if not voice_path.exists():
        print(f"voice audio not found: {voice_path}")
        sys.exit(3)
    sys.exit(run(face_path, voice_path))


if __name__ == "__main__":
    main()
