#!/usr/bin/env python
"""Phase 4 - audio augmentation and feature extraction.

Reads voice clips under ``audio_data/<member>/*.wav``, creates augmented copies,
extracts a fixed feature vector per clip, and writes them to
``data/processed/audio_features.csv``.

Public interface for Phase 6 (app.py):
    from audio_features import extract_features_for_file, FEATURE_COLUMNS

Run:
    python scripts/audio_features.py            # build the feature table
    python scripts/audio_features.py --no-aug   # originals only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# Paths & constants

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "audio_data"
OUT_CSV = ROOT / "data" / "processed" / "audio_features.csv"

SR = 16_000          # load / resample every clip to 16 kHz mono
N_MFCC = 13
TRIM_DB = 25         # trim silence quieter than this

MEMBERS = [
    "Gentil_Tonny_Christian_Iradukunda",
    "Hassan_Adelani_Luqman",
    "Mahlet_Assefa_Tilahun",
    "Yvette_Uwimpaye",
]

# Non-feature columns (excluded by the model). `augmented` is the boolean flag
# PLAN.md asks for; `variant` keeps the specific kind (original/pitch_up/...).
META_COLUMNS = ["member", "phrase", "source_clip_id", "variant", "augmented"]

# Canonical, ordered feature columns. The CSV and the model both rely on this order.
FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["rolloff_mean", "rolloff_std",
       "rms_mean", "rms_std",
       "zcr_mean", "zcr_std",
       "centroid_mean", "centroid_std"]
)



# Helpers

def detect_phrase(stem: str) -> str:
    """Map a filename to one of the two known phrases."""
    s = stem.lower()
    if "approve" in s:
        return "yes_approve"
    if "confirm" in s or "transaction" in s:
        return "confirm_transaction"
    return "unknown_phrase"


def load_clip(path: Path) -> np.ndarray:
    """Load a WAV as mono 16 kHz and trim silence."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    return y


def augment(y: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    """Return augmented versions of a signal, keyed by variant name."""
    rng = np.random.default_rng(0)  # fixed seed -> reproducible noise
    noise = 0.005 * rng.standard_normal(len(y))
    return {
        "pitch_up":     librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "time_stretch": librosa.effects.time_stretch(y, rate=0.9),
        "noise":        y + noise,
    }


def extract_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Turn a signal into the fixed dict of features in FEATURE_COLUMNS."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)          # (13, frames)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)          # (1, frames)
    rms = librosa.feature.rms(y=y)                                  # (1, frames)
    zcr = librosa.feature.zero_crossing_rate(y)                     # (1, frames)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)        # (1, frames)

    feats: dict[str, float] = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())
    feats["rolloff_mean"] = float(rolloff.mean())
    feats["rolloff_std"] = float(rolloff.std())
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())
    feats["centroid_mean"] = float(centroid.mean())
    feats["centroid_std"] = float(centroid.std())
    return feats


def extract_features_for_file(path: str | Path) -> dict[str, float]:
    """Path in -> feature dict out (no augmentation). Used by app.py."""
    y = load_clip(Path(path))
    return extract_features(y, SR)



# Build the feature table

def build_table(save_aug: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    n_sources = 0

    for member in MEMBERS:
        folder = AUDIO_DIR / member
        if not folder.is_dir():
            continue
        wavs = sorted(p for p in folder.glob("*.wav") if "_aug_" not in p.name)
        for wav in wavs:
            phrase = detect_phrase(wav.stem)
            source_clip_id = f"{member}__{phrase}"
            n_sources += 1

            y = load_clip(wav)

            # original
            rows.append({
                "member": member, "phrase": phrase,
                "source_clip_id": source_clip_id,
                "variant": "original", "augmented": False,
                **extract_features(y, SR),
            })

            # augmented copies (share the parent's source_clip_id)
            for variant, y_aug in augment(y, SR).items():
                rows.append({
                    "member": member, "phrase": phrase,
                    "source_clip_id": source_clip_id,
                    "variant": variant, "augmented": True,
                    **extract_features(y_aug, SR),
                })
                if save_aug:  # write augmented WAV (gitignored) for inspection
                    out = wav.with_name(f"{wav.stem}_aug_{variant}.wav")
                    sf.write(out, y_aug, SR)

    if not rows:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS)

    df = pd.DataFrame(rows)[META_COLUMNS + FEATURE_COLUMNS]
    print(f"Processed {n_sources} source clip(s) -> {len(df)} rows "
          f"({len(df) - n_sources} augmented).")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build audio_features.csv")
    ap.add_argument("--no-aug", action="store_true",
                    help="originals only; skip augmentation")
    args = ap.parse_args()

    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*/*.wav")):
        print("No .wav files found under audio_data/<member>/.")
        print("Add clips (or run scripts/convert_audio.py to make WAVs from phone")
        print("recordings) and re-run. Expected: audio_data/<member>/"
              "<member>_yes_approve.wav and ..._confirm_transaction.wav")
        return

    df = build_table(save_aug=not args.no_aug)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}  ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
