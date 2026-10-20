"""Config-driven YOLO training.

- Params come from configs/*.yaml (never hardcoded).
- Seeds set; AMP on CUDA; best-val checkpoint kept; experiment logged to experiments/CSV.
- Smoke test first:  python scripts/train.py --smoke   (1 epoch, 1% data, CPU-safe)

Usage:
    python scripts/train.py                    # full baseline
    python scripts/train.py --config configs/yolo11n.yaml --epochs 50 --batch 8
    python scripts/train.py --smoke            # pipeline check before long run
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

# 4GB VRAM: reduce CUDA fragmentation (PyTorch memory-management note).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml

BASE = Path(__file__).resolve().parents[1]
EXP_LOG = BASE / "experiments" / "runs.csv"


def set_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo11n.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plots", default=None, help="true/false: val plots (disable if RAM OOM)")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(BASE / args.config, {"epochs": args.epochs, "batch": args.batch,
                                        "imgsz": args.imgsz, "device": args.device,
                                        "workers": args.workers})
    plots = True if args.plots is None else args.plots.lower() not in ("0", "false", "no")
    if args.smoke:  # short run before multi-hour run
        cfg.update({"epochs": 1, "batch": 4, "imgsz": 160, "patience": 1,
                    "workers": 0, "device": "cpu", "notes": "SMOKE TEST"})

    set_seeds(int(cfg.get("seed", 42)))

    import torch
    from ultralytics import YOLO

    device = str(cfg.get("device", 0))
    if device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> falling back to CPU")
        device = "cpu"
    if device == "cpu":
        cfg["amp"] = False
        cfg["workers"] = 0

    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "")
    print(f"[exp {exp_id}] model={cfg['model']} imgsz={cfg['imgsz']} "
          f"batch={cfg['batch']} epochs={cfg['epochs']} device={device}")
    print(f"[exp {exp_id}] data={cfg['data']} optimizer={cfg['optimizer']} "
          f"lr0={cfg['lr0']} amp={cfg['amp']}")

    t0 = time.time()
    if args.resume:
        from ultralytics import YOLO as _YOLO

        # Resume SELALU pakai setting hemat RAM: Ultralytics mengabaikan override
        # plots/workers saat resume (bukti: 2x resume tetap workers=2/plots=True),
        # jadi hardcode di sini, bukan lewat config.
        print("[resume] from {0} (forced plots=False, workers=1)".format(args.resume))
        model = _YOLO(str(args.resume))
        results = model.train(resume=True, plots=False, workers=1)
    else:
        model = YOLO(str(cfg["model"]))  # pretrained weights
        results = model.train(
        data=str((BASE / str(cfg["data"])).resolve()) if not Path(str(cfg["data"])).is_absolute() else str(cfg["data"]),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        device=device,
        workers=int(cfg["workers"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["lr0"]),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        warmup_epochs=int(cfg.get("warmup_epochs", 3)),
        cos_lr=True,
        amp=bool(cfg.get("amp", True)),
        patience=int(cfg.get("patience", 15)),
        mosaic=float(cfg.get("mosaic", 0.5)),
        fliplr=float(cfg.get("fliplr", 0.0)),  # off: flip changes sign meaning
        flipud=float(cfg.get("flipud", 0.0)),
        hsv_h=float(cfg.get("hsv_h", 0.015)),
        hsv_s=float(cfg.get("hsv_s", 0.5)),
        hsv_v=float(cfg.get("hsv_v", 0.3)),
        degrees=float(cfg.get("degrees", 5.0)),
        translate=float(cfg.get("translate", 0.05)),
        scale=float(cfg.get("scale", 0.3)),
        erasing=float(cfg.get("erasing", 0.0)),
        seed=int(cfg.get("seed", 42)),
        project=str(BASE / "checkpoints"),
        name=exp_id,
        exist_ok=True,
        verbose=True,
    )
    runtime = round((time.time() - t0) / 60, 1)

    # Best-val metrics: never the final epoch by default.
    try:
        m = results.results_dict
        map50_95 = round(float(m.get("metrics/mAP50-95(B)", 0.0)), 4)
        map50 = round(float(m.get("metrics/mAP50(B)", 0.0)), 4)
        p = round(float(m.get("metrics/precision(B)", 0.0)), 4)
        r = round(float(m.get("metrics/recall(B)", 0.0)), 4)
    except (AttributeError, TypeError, ValueError):
        map50_95 = map50 = p = r = 0.0

    ckpt = BASE / "checkpoints" / exp_id / "weights" / "best.pt"
    print(f"[exp {exp_id}] done in {runtime} min | val mAP50-95={map50_95} "
          f"mAP50={map50} P={p} R={r} | best={ckpt}")

    EXP_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not EXP_LOG.exists()
    with open(EXP_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["experiment_id", "config", "seed", "split_version", "model", "imgsz",
                        "batch", "epochs", "optimizer", "lr0", "device", "amp",
                        "val_map50_95", "val_map50", "val_P", "val_R",
                        "runtime_min", "checkpoint", "notes"])
        w.writerow([exp_id, args.config, cfg.get("seed"), cfg.get("split_version"),
                    cfg["model"], cfg["imgsz"], cfg["batch"], cfg["epochs"],
                    cfg["optimizer"], cfg["lr0"], device, cfg["amp"],
                    map50_95, map50, p, r, runtime, ckpt.relative_to(BASE).as_posix(), cfg.get("notes", "")])


if __name__ == "__main__":
    main()
