"""Group-aware re-split v2.

Problem (proven by check_near_dup.py): the raw train/val split was made AFTER
augmentation, so 63% of val images are pHash near-copies of train images.
Val mAP 0.994 on v1 is therefore inflated.

Fix: cluster ALL cleaned images by pHash (dist <= 6 = same acquisition group),
then assign WHOLE groups to train/val/test, stratified per class with target
ratios 80/12/8. No group spans splits. Split version bumped to v2.

Reads : data/raw/ (raw, never modified)
Writes: data/processed/{images,labels}/{train,val,test} (overwritten, v2)
        data/splits/*_v2.txt, data/bisindo.yaml (unchanged schema), reports

Usage: python scripts/make_group_splits.py [--thr 6] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
SPLITS_DIR = BASE / "data" / "splits"
REPORTS = BASE / "data" / "reports"
CLASSES = [chr(ord("A") + i) for i in range(26)]
IMG_EXTS = (".jpg", ".jpeg", ".png")
TARGET = {"train": 0.80, "val": 0.12, "test": 0.08}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def valid_label(txt: Path) -> bool:
    try:
        lines = txt.read_text().strip().splitlines()
    except OSError:
        return False
    if not lines or not lines[0].strip():
        return False
    for ln in lines:
        p = ln.split()
        if len(p) != 5:
            return False
        try:
            if not (0 <= int(float(p[0])) <= 25):
                return False
            if any(not (0.0 <= float(v) <= 1.0) for v in p[1:]):
                return False
        except ValueError:
            return False
    return True


def popcount(a: np.ndarray) -> np.ndarray:
    d = np.zeros(len(a), dtype=np.int32)
    for s in range(64):
        d += ((a >> np.uint64(s)) & np.uint64(1)).astype(np.int32)
    return d


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # --- 1. collect + clean ---
    pooled: list[tuple[Path, Path, int]] = []  # (img, label, class_idx)
    for split in ("train", "val"):
        for cls in CLASSES:
            img_dir, lbl_dir = RAW / "images" / split / cls, RAW / "labels" / split / cls
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                    continue
                txt = lbl_dir / (img.stem + ".txt")
                if txt.exists() and valid_label(txt):
                    pooled.append((img, txt, int(float(txt.read_text().split()[0]))))
    # exact-md5 dedupe globally
    seen: dict[str, tuple[Path, Path, int]] = {}
    for img, txt, ci in sorted(pooled, key=lambda t: str(t[0])):
        h = md5(img)
        if h not in seen:
            seen[h] = (img, txt, ci)
    pooled = list(seen.values())
    print(f"pooled cleaned: {len(pooled)} images")

    # --- 2. pHash + cluster (union-find, thr Hamming) ---
    print("hashing...")
    hashes = []
    for img, _, _ in pooled:
        with Image.open(img) as im:
            hashes.append(int(str(imagehash.phash(im)), 16))
    H = np.array(hashes, dtype=np.uint64)
    n = len(H)
    # Cluster WITHIN each class only: same background across DIFFERENT letters is
    # fine (model must distinguish letters anyway); same-letter near-copies across
    # splits are the leakage. Cross-class clustering over-groups and breaks strat.
    idx_by_class: dict[int, list[int]] = defaultdict(list)
    for i, (_, _, ci) in enumerate(pooled):
        idx_by_class[ci].append(i)
    uf = UnionFind(n)
    for ci, idxs in idx_by_class.items():
        Hs = H[idxs]
        for a, i in enumerate(idxs):
            d = popcount(np.bitwise_xor(Hs, Hs[a]))
            for b in np.where(d <= args.thr)[0].tolist():
                j = idxs[b]
                if j > i:
                    uf.union(i, j)
        print(f"  class {CLASSES[ci]}: {len(idxs)} imgs clustered")
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"groups: {len(groups)}, largest: {sizes[:5]}")

    # --- 3. stratified group assignment to 80/12/8 ---
    by_class: dict[int, list[list[int]]] = defaultdict(list)
    for g in groups.values():
        ci = pooled[g[0]][2]
        by_class[ci].append(g)
    assign: dict[int, str] = {}
    totals = {"train": 0, "val": 0, "test": 0}
    for ci in sorted(by_class):
        gs = by_class[ci]
        rng.shuffle(gs)
        want = {s: round(sum(len(g) for g in gs) * TARGET[s]) for s in TARGET}
        want["train"] = sum(len(g) for g in gs) - want["val"] - want["test"]
        cur = {"train": 0, "val": 0, "test": 0}
        for g in sorted(gs, key=len, reverse=True):
            # put group where deficit is largest
            s = max(TARGET, key=lambda k: want[k] - cur[k])
            for i in g:
                assign[i] = s
            cur[s] += len(g)
            totals[s] += len(g)

    # --- 4. write v2 outputs ---
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    names: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for i, (img, txt, _) in enumerate(pooled):
        s = assign[i]
        flat = f"{img.parent.name}_{img.name}"
        (PROCESSED / "images" / s).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, PROCESSED / "images" / s / flat)
        shutil.copy2(txt, PROCESSED / "labels" / s / (Path(flat).stem + ".txt"))
        names[s].append(flat)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for s, ns in names.items():
        (SPLITS_DIR / f"{s}_v2.txt").write_text("\n".join(sorted(ns)))
    (BASE / "data" / "bisindo.yaml").write_text(
        "# GENERATED by make_group_splits.py (v2) — do not hand-edit\n"
        "# No path key: Ultralytics resolves the splits relative to this file.\n"
        "train: processed/images/train\nval: processed/images/val\ntest: processed/images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
    )
    report = {
        "split_version": "v2", "phash_thr": args.thr, "seed": args.seed,
        "pooled": len(pooled), "groups": len(groups),
        "largest_groups": sizes[:10],
        "final": {s: len(names[s]) for s in names},
    }
    (REPORTS / "split_report_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
