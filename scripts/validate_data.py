#!/usr/bin/env python
"""Phase 1 - data ingestion & validation.

Confirms the tabular CSVs load, that each member has the expected media, that
every present file is non-empty and actually decodable, and prints a readiness
report. Media the team has explicitly agreed to deliver later is listed in
KNOWN_PENDING and reported as PENDING rather than as a hard failure.

Run:   python scripts/validate_data.py
Exit:  0 if ready (or only known-pending items remain); 1 if there are
       unexpected problems (undecodable files, or missing data not agreed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import librosa
import pandas as pd
import PIL.Image as Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG = ROOT / "image_data"
AUD = ROOT / "audio_data"

MEMBERS = [
    "Gentil_Tonny_Christian_Iradukunda",
    "Hassan_Adelani_Luqman",
    "Mahlet_Assefa_Tilahun",
    "Yvette_Uwimpaye",
]
EXPECTED_IMAGES = 3
EXPECTED_AUDIO = 2  # counted as WAVs (the format the pipeline consumes)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Media the team has agreed to deliver in a later phase. Keyed by (modality, member).
# Items here are reported as PENDING, not as failures. Add to this dict only after
# the team explicitly agrees a gap is deferred.
KNOWN_PENDING = {
    ("image", "Yvette_Uwimpaye"): "agreed: provided in Phase 3",
}

CSVS = {
    "customer_social_profiles": RAW / "customer_social_profiles.csv",
    "customer_transactions": RAW / "customer_transactions.csv",
}


def media_files(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts and p.name != ".gitkeep"
    )


def image_ok(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size == 0:
            return False, "empty file"
        im = Image.open(path)
        im.load()
        return True, f"{im.size[0]}x{im.size[1]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


def audio_ok(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size == 0:
            return False, "empty file"
        y, sr = librosa.load(path, sr=None)
        return True, f"{len(y) / sr:.1f}s@{sr}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


def check_csvs() -> bool:
    print("=" * 70)
    print("TABULAR DATA")
    print("=" * 70)
    ok = True
    for name, path in CSVS.items():
        print(f"\n[{name}]  {path.relative_to(ROOT)}")
        if not path.exists():
            print("  MISSING")
            ok = False
            continue
        df = pd.read_csv(path)
        print(f"  shape       : {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  columns     : {list(df.columns)}")
        nulls = df.isna().sum()
        nulls = nulls[nulls > 0]
        print(f"  null counts : {dict(nulls) if len(nulls) else 'none'}")
        dups = int(df.duplicated().sum())
        print(f"  duplicate rows : {dups}")
        print(f"  head:\n{df.head(3).to_string(max_colwidth=24)}")
    return ok


def check_media() -> tuple[bool, bool]:
    """Returns (has_hard_problem, has_pending)."""
    print("\n" + "=" * 70)
    print("MEDIA DATA (per member)")
    print("=" * 70)
    print(f"\n{'member':<34}{'images':<16}{'audio (wav)':<16}{'status'}")
    print("-" * 82)

    hard_problem = False
    has_pending = False

    for m in MEMBERS:
        imgs = media_files(IMG / m, IMG_EXTS)
        wavs = media_files(AUD / m, {".wav"})

        bad_imgs = [p.name for p in imgs if not image_ok(p)[0]]
        bad_wavs = [p.name for p in wavs if not audio_ok(p)[0]]

        img_short = len(imgs) < EXPECTED_IMAGES
        aud_short = len(wavs) < EXPECTED_AUDIO

        notes = []
        status = "OK"

        # images
        if img_short:
            if ("image", m) in KNOWN_PENDING:
                has_pending = True
                status = "PENDING"
                notes.append(f"images {KNOWN_PENDING[('image', m)]}")
            else:
                hard_problem = True
                status = "MISSING"
                notes.append(f"images {len(imgs)}/{EXPECTED_IMAGES}")
        if bad_imgs:
            hard_problem = True
            status = "BAD FILE"
            notes.append(f"undecodable img: {bad_imgs}")

        # audio
        if aud_short:
            if ("audio", m) in KNOWN_PENDING:
                has_pending = True
                if status == "OK":
                    status = "PENDING"
                notes.append(f"audio {KNOWN_PENDING[('audio', m)]}")
            else:
                hard_problem = True
                if status in ("OK", "PENDING"):
                    status = "MISSING"
                notes.append(f"audio {len(wavs)}/{EXPECTED_AUDIO}")
        if bad_wavs:
            hard_problem = True
            status = "BAD FILE"
            notes.append(f"undecodable audio: {bad_wavs}")

        icell = f"{len(imgs)}/{EXPECTED_IMAGES}" + (" !" if bad_imgs else "")
        acell = f"{len(wavs)}/{EXPECTED_AUDIO}" + (" !" if bad_wavs else "")
        print(f"{m:<34}{icell:<16}{acell:<16}{status}")
        for n in notes:
            print(f"{'':<34}-> {n}")

    # impostor
    imp = media_files(IMG / "impostor", IMG_EXTS)
    bad_imp = [p.name for p in imp if not image_ok(p)[0]]
    print("-" * 82)
    imp_status = "OK" if imp and not bad_imp else ("BAD FILE" if bad_imp else "MISSING")
    print(f"{'impostor':<34}{str(len(imp)) + ' file(s)':<32}{imp_status}")
    if not imp:
        hard_problem = True
    if bad_imp:
        hard_problem = True
        print(f"{'':<34}-> undecodable: {bad_imp}")

    return hard_problem, has_pending


def main() -> int:
    csv_ok = check_csvs()
    hard_problem, has_pending = check_media()
    hard_problem = hard_problem or not csv_ok

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if hard_problem:
        print("NOT READY - unexpected gaps or unreadable files above (not agreed-pending).")
        print("Per the plan, resolve these before building on the data.")
    elif has_pending:
        print("READY except for agreed-pending items (listed PENDING above).")
        print("Downstream phases may start; pending media must land before its phase.")
    else:
        print("READY - all data present, readable, and complete.")

    if has_pending:
        print("\nAgreed-pending:")
        for (modality, member), reason in KNOWN_PENDING.items():
            print(f"  - {member} {modality}: {reason}")

    return 1 if hard_problem else 0


if __name__ == "__main__":
    sys.exit(main())
