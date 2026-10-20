"""Baseline: frozen DINOv2 ViT-S/14 image embeddings + RBF SVM.

Same shape as the landmark baseline (frozen features + SVM head) but the features are a
general-purpose image embedding of the whole photo, background included. Same splits and
one-answer-per-photo rule as the detectors.

Usage: python scripts/probe_dino.py   # about 10 minutes on a 4 GB GPU
Writes: experiments/dino_probe/{train,val,holdout}.npz + summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from PIL import Image
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0).to(device).eval()
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg)

    # The embedding sees the whole photo, background included; batch 16 fits a 4 GB GPU.
    def embed(split: str):
        src = (BASE / "data/holdout_achmadnoer" if split == "holdout"
               else BASE / "data/processed/images" / split)
        imgs = sorted([p for p in src.rglob("*.jpg")] + [p for p in src.rglob("*.png")])
        X, y = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), 16):
                batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in imgs[i:i + 16]]).to(device)
                X.append(model(batch).cpu().numpy())
                for p in imgs[i:i + 16]:
                    lab = p.parent.name if split == "holdout" else p.name.split("_")[0]
                    y.append(LETTERS.index(lab))
        return np.concatenate(X), np.array(y)

    out = BASE / "experiments/dino_probe"
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for s in ["train", "val", "holdout"]:
        t0 = time.perf_counter()
        X, y = embed(s)
        np.savez_compressed(out / f"{s}.npz", X=X, y=y)
        data[s] = (X, y)
        print(s, X.shape, f"{time.perf_counter()-t0:.0f}s", flush=True)

    Xtr, ytr = data["train"]
    # shortcut: 3x3 grid, 3-fold, one seed; a wider grid did not move the landmark baseline either.
    gs = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
        {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
        cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=-1)
    gs.fit(Xtr, ytr)
    rep = {"best": gs.best_params_, "cv": round(float(gs.best_score_), 4)}
    for s in ["val", "holdout"]:
        X, y = data[s]
        ok = int((gs.predict(X) == y).sum())
        rep[s] = round(ok / len(y), 4)
    (out / "summary.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
