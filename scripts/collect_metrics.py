#!/usr/bin/env python
"""Phase 6 - collect the per-model metric shards into one results/metrics.json.

Each model writes its own shard (results/metrics/<name>.json) so the three parallel
phases never conflict on a shared file (PLAN.md 3.7). This is the ONLY script that writes
results/metrics.json, which the Phase 8 report reads as the single source of truth.

Run:
    python scripts/collect_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "metrics.json"


def main() -> None:
    shards = {}
    for path in sorted(SHARD_DIR.glob("*.json")):
        shards[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not shards:
        print(f"No metric shards found in {SHARD_DIR.relative_to(ROOT)}")
        return
    OUT.write_text(json.dumps(shards, indent=2), encoding="utf-8")
    print(f"Collected {len(shards)} shard(s) -> {OUT.relative_to(ROOT)}")
    for name in shards:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
