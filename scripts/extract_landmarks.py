"""Baseline features: MediaPipe Hands landmarks for the landmark + SVM baseline.

xyz: the 21 landmarks of the first detected hand, relative to the wrist and divided by the
largest absolute coordinate (63 values). geo: 15 fingertip/wrist distances + 15 joint-angle
cosines (30 values). Photos where no hand is found are listed and count as wrong later.
Only one hand is tracked, while many BISINDO letters use two.

Reads : data/processed/images/{train,val} | data/holdout_achmadnoer
Writes: experiments/landmarks/{split}.npz (X, y, files) + report.json

Usage:
    python scripts/extract_landmarks.py --split val
    python scripts/extract_landmarks.py --split train
    python scripts/extract_landmarks.py --split holdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def features_from_landmarks(lm) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
    pts -= pts[0]  # wrist-relative
    s = np.abs(pts).max() or 1.0
    return (pts / s).reshape(-1)


def features_geo(lm) -> np.ndarray:
    """Rotation, translation and scale invariant: 15 pairwise distances between the wrist and
    the five fingertips (scale-normalised) + 15 joint-flexion cosines (3 per finger) = 30-D."""
    p = np.array([[q.x, q.y, q.z] for q in lm.landmark], dtype=np.float32)
    tips, wrist = [4, 8, 12, 16, 20], p[0]
    d = [np.linalg.norm(p[i] - p[j]) for a, i in enumerate([0] + tips)
         for j in ([0] + tips)[a + 1:]]
    d = np.array(d, np.float32)
    d /= d.max() or 1.0
    chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
              [13, 14, 15, 16], [17, 18, 19, 20]]
    ang = []
    for ch in chains:
        for a, b, c in zip(ch, ch[1:], ch[2:]):
            u, v = p[a] - p[b], p[c] - p[b]
            ang.append(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    return np.concatenate([d, np.array(ang, np.float32)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "holdout"])
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--feat", default="xyz", choices=["xyz", "geo"])
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from tqdm import tqdm

    src = (BASE / "data" / "holdout_achmadnoer" if args.split == "holdout"
           else BASE / "data" / "processed" / "images" / args.split)
    imgs = sorted([p for p in src.rglob("*.jpg")] + [p for p in src.rglob("*.png")])
    assert imgs, f"no images in {src}"

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=args.min_conf)
    X, y, files, fails = [], [], [], []
    for p in tqdm(imgs, desc=f"landmarks/{args.split}"):
        # label: letter folder (holdout) or file-name prefix (processed)
        if args.split == "holdout":
            label = p.parent.name
        else:
            label = p.name.split("_")[0]
        r = hands.process(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        if not r.multi_hand_landmarks:
            fails.append(p.name)
            continue
        feat = features_geo(r.multi_hand_landmarks[0]) if args.feat == "geo" \
            else features_from_landmarks(r.multi_hand_landmarks[0])
        X.append(feat)
        y.append(LETTERS.index(label))
        files.append(p.name)
    hands.close()

    out = BASE / "experiments" / "landmarks"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.split if args.feat == "xyz" else f"{args.split}_{args.feat}"
    np.savez_compressed(out / f"{tag}.npz", X=np.array(X), y=np.array(y),
                        files=np.array(files))
    rep = {"split": args.split, "feat": args.feat, "n_images": len(imgs), "n_ok": len(X),
           "n_fail": len(fails), "fail_rate": round(len(fails) / len(imgs), 4)}
    (out / f"{tag}_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
