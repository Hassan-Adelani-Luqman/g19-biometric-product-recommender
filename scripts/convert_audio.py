#!/usr/bin/env python
"""Convert phone-recorded audio (.aac/.m4a/...) to 16 kHz mono WAV.

Phase 1 ingestion helper. The clips the team recorded are AAC/M4A, which
libsndfile (soundfile) cannot read and which otherwise need a system ffmpeg
that isn't installed. PyAV bundles its own ffmpeg, so we decode with PyAV and
write clean PCM-16 WAVs that librosa/soundfile read natively. Originals are
kept for provenance; WAVs are written alongside them in audio_data/<member>/.

Usage:
    python scripts/convert_audio.py            # convert anything missing a .wav
    python scripts/convert_audio.py --force    # re-convert even if the .wav exists
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
from av.audio.resampler import AudioResampler
import numpy as np
import soundfile as sf

AUDIO_ROOT = Path(__file__).resolve().parents[1] / "audio_data"
SRC_EXTS = {".aac", ".m4a", ".mp3", ".ogg", ".flac", ".mp4", ".wma"}
TARGET_SR = 16000


def convert_one(src: Path, dst: Path, sr: int = TARGET_SR) -> float:
    """Decode `src` and write a mono `sr`-Hz PCM-16 WAV to `dst`. Returns duration (s)."""
    resampler = AudioResampler(format="s16", layout="mono", rate=sr)
    chunks: list[np.ndarray] = []
    with av.open(str(src)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                chunks.append(rframe.to_ndarray())
        for rframe in resampler.resample(None):  # flush the resampler
            chunks.append(rframe.to_ndarray())
    if not chunks:
        raise RuntimeError("no audio frames decoded")
    data = np.concatenate(chunks, axis=1).reshape(-1).astype(np.int16)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return len(data) / sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-convert even if the WAV exists")
    args = ap.parse_args()

    sources = sorted(p for p in AUDIO_ROOT.rglob("*") if p.suffix.lower() in SRC_EXTS)
    if not sources:
        print(f"No source audio found under {AUDIO_ROOT}")
        return

    n_done = n_skip = n_fail = 0
    for src in sources:
        dst = src.with_suffix(".wav")
        rel = src.relative_to(AUDIO_ROOT.parent)
        if dst.exists() and not args.force:
            print(f"  skip  {rel} (WAV exists)")
            n_skip += 1
            continue
        try:
            dur = convert_one(src, dst)
            print(f"  ok    {rel} -> {dst.name}  ({dur:.2f}s)")
            n_done += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL  {rel}  {type(exc).__name__}: {exc}")
            n_fail += 1

    print(f"\nConverted {n_done}, skipped {n_skip}, failed {n_fail}.")


if __name__ == "__main__":
    main()
