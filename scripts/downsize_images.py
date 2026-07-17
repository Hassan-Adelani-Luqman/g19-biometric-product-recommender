#!/usr/bin/env python
"""Downsize large source photos in place (cap longest side, bake EXIF orientation).

Phone cameras produce multi-MB images; the face pipeline only needs a modest
resolution (Phase 3 detects, crops, and downscales the face anyway). This caps
the longest side at --max-side and re-encodes, and bakes in EXIF orientation so
downstream tools (which ignore EXIF) see upright faces. Images already within
--max-side are left byte-for-byte untouched, so the script is idempotent.

Usage:  python scripts/downsize_images.py [--max-side 1280] [--quality 87]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

IMG_ROOT = Path(__file__).resolve().parents[1] / "image_data"
EXTS = {".jpg", ".jpeg", ".png"}


def process(path: Path, max_side: int, quality: int) -> tuple[str, int, int]:
    """Resize `path` in place if it exceeds max_side. Returns (message, before, after)."""
    im = Image.open(path)
    w, h = im.size
    before = path.stat().st_size
    if max(w, h) <= max_side:
        return f"skip  {path.name} ({w}x{h}, already <= {max_side}px)", before, before

    im = ImageOps.exif_transpose(im)  # bake orientation, then drop EXIF on save
    fmt = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    if fmt == "JPEG":
        im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    save_kwargs = {"quality": quality, "optimize": True} if fmt == "JPEG" else {"optimize": True}
    im.save(path, fmt, **save_kwargs)
    after = path.stat().st_size
    msg = f"ok    {path.name}  {w}x{h} -> {im.size[0]}x{im.size[1]}   {before // 1024}KB -> {after // 1024}KB"
    return msg, before, after


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--quality", type=int, default=87)
    args = ap.parse_args()

    paths = sorted(p for p in IMG_ROOT.rglob("*") if p.suffix.lower() in EXTS)
    total_before = total_after = 0
    for p in paths:
        msg, before, after = process(p, args.max_side, args.quality)
        total_before += before
        total_after += after
        print("  " + msg)
    print(f"\nTotal image_data: {total_before // 1024}KB -> {total_after // 1024}KB")


if __name__ == "__main__":
    main()
