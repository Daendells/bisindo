"""Near-duplicate audit, train-vs-val.

pHash (64-bit) per image; for each val image, min Hamming distance to any train
image. Reports distribution + lists worst offenders for manual inspection.
Threshold guide: 0 = identical, <=6 = near-duplicate/augmented copy, >10 = distinct.

Usage: python scripts/check_near_dup.py  (takes a few minutes on CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed" / "images"
OUT = BASE / "data" / "reports" / "near_dup_audit.json"


def phashes(split: str) -> tuple[list[str], np.ndarray]:
    names, hashes = [], []
    for p in sorted((PROC / split).iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                with Image.open(p) as im:
                    h = imagehash.phash(im)
                names.append(p.name)
                hashes.append(int(str(h), 16))
            except (OSError, ValueError):
                pass
    arr = np.array(hashes, dtype=np.uint64)
    return names, arr


def min_dists(val: np.ndarray, train: np.ndarray) -> np.ndarray:
    out = np.empty(len(val), dtype=np.int32)
    for i, v in enumerate(val):
        x = np.bitwise_xor(train, np.uint64(v))
        d = np.zeros(len(x), dtype=np.int32)
        for shift in range(64):
            d += ((x >> np.uint64(shift)) & np.uint64(1)).astype(np.int32)
        out[i] = int(d.min())
        if (i + 1) % 400 == 0:
            print(f"  ...{i + 1}/{len(val)}")
    return out


def main() -> None:
    print("hashing train...")
    _, train = phashes("train")
    print(f"train: {len(train)} hashes")
    print("hashing val...")
    val_names, val = phashes("val")
    print(f"val: {len(val)} hashes")
    print("comparing...")
    dists = min_dists(val, train)
    report = {
        "train_n": len(train),
        "val_n": len(val),
        "val_min_dist_mean": round(float(dists.mean()), 2),
        "val_min_dist_median": float(np.median(dists)),
        "frac_le_2": round(float((dists <= 2).mean()), 4),
        "frac_le_6": round(float((dists <= 6).mean()), 4),
        "frac_le_10": round(float((dists <= 10).mean()), 4),
        "worst_20": sorted(
            ((n, int(d)) for n, d in zip(val_names, dists)), key=lambda t: t[1]
        )[:20],
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "worst_20"}, indent=2))
    print("worst 10:", report["worst_20"][:10])


if __name__ == "__main__":
    main()
