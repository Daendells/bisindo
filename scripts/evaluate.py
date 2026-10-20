"""Reproducible evaluation + error analysis.

Loads best.pt ONCE, reproduces val (and --split test ONLY for final reporting, S1.3):
  metrics, confusion matrix, per-class report, prediction CSV, error tables.

Usage:
    python scripts/evaluate.py --weights checkpoints/<exp>/weights/best.pt
    python scripts/evaluate.py --weights checkpoints/<exp>/weights/best.pt --split test   # FINAL only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the checkpoint was trained at")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.split == "test":
        print("[guard] TEST touched: only valid AFTER model selection is frozen.")

    from ultralytics import YOLO

    ckpt = (BASE / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    out_dir = BASE / "experiments" / (ckpt.parents[1].name + f"_{args.split}_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ckpt))
    args.imgsz = args.imgsz or int(model.ckpt["train_args"]["imgsz"])
    metrics = model.val(data=str((BASE / "data" / "bisindo.yaml").resolve()),
                        split=args.split, imgsz=args.imgsz, plots=True, save_json=True)

    det = metrics.box
    summary = {
        "weights": ckpt.relative_to(BASE).as_posix(),
        "split": args.split,
        "map50-95": round(float(det.map), 4),
        "map50": round(float(det.map50), 4),
        "mean_precision": round(float(det.mp), 4),
        "mean_recall": round(float(det.mr), 4),
        "fitness": round(float(det.fitness()), 4),
    }
    print(summary)

    # Per-image predictions for error analysis: rank high-conf mistakes manually.
    results = model.predict(source=str(BASE / "data" / "processed" / "images" / args.split),
                            imgsz=args.imgsz, conf=args.conf, verbose=False)
    rows = []
    for res in results:
        img = Path(str(res.path)).name
        if res.boxes is None or len(res.boxes) == 0:
            rows.append([img, "", "", "", "no-pred"])
            continue
        for box in res.boxes:
            rows.append([img, int(box.cls), round(float(box.conf), 3),
                         [round(float(v), 3) for v in box.xywhn[0].tolist()], "pred"])
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "pred_cls", "conf", "xywhn", "flag"]] + rows)

    (out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    print(f"[eval] artifacts -> {out_dir}")
    print("[next] open confusion_matrix.png, sort predictions.csv by conf, "
          "inspect top confusions side-by-side before any tuning.")


if __name__ == "__main__":
    main()
