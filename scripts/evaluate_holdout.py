"""Unseen-signer evaluation: YOLO top-1 against the letter folder (no boxes needed).

Set: achmadnoer/alfabet-bisindo (CC0), 312 photos = 26 letters x 12, three backgrounds,
a different signer and camera from the training data. The truth is the folder name.
One answer per photo (the most confident box); no box counts as wrong.

Usage:
    python scripts/evaluate_holdout.py --weights checkpoints/<exp>/weights/best.pt
    python scripts/evaluate_holdout.py --weights <...> --src data/holdout_achmadnoer
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default="data/holdout_achmadnoer")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the checkpoint was trained at")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt = args.weights if Path(args.weights).is_absolute() else str(BASE / args.weights)
    src = BASE / args.src
    imgs = sorted([p for p in src.rglob("*.jpg")] + [p for p in src.rglob("*.png")])
    assert imgs, f"no images in {src}"

    model = YOLO(ckpt)
    args.imgsz = args.imgsz or int(model.ckpt["train_args"]["imgsz"])
    out_dir = BASE / "experiments" / (Path(ckpt).parents[1].name + "_external_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    correct, rows, err = 0, [], Counter()
    for img in imgs:
        true = img.parent.name  # the letter folder
        res = model.predict(source=str(img), imgsz=args.imgsz,
                            conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            rows.append([img.name, true, "", "", "no-pred"])
            continue
        best = max(res.boxes, key=lambda b: float(b.conf))
        pred = LETTERS[int(best.cls)]
        conf = round(float(best.conf), 3)
        ok = pred == true
        correct += ok
        if not ok:
            err[(true, pred)] += 1
        rows.append([img.name, true, pred, conf, "ok" if ok else "wrong"])

    n = len(imgs)
    summary = {
        "weights": Path(ckpt).resolve().relative_to(BASE).as_posix(),
        "n": n,
        "top1_acc": round(correct / n, 4),
        "n_nopred": sum(1 for r in rows if r[4] == "no-pred"),
        "top_confusions": [[a, b, c] for (a, b), c in err.most_common(10)],
    }
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "true", "pred", "conf", "flag"]] + rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
