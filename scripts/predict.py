"""Single-image / webcam-spotcheck inference (same preprocessing as training)."""
from __future__ import annotations

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file/dir or 0 for webcam")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt = args.weights if Path(args.weights).is_absolute() else str(BASE / args.weights)
    src = args.source if args.source == "0" else (
        args.source if Path(args.source).is_absolute() else str(BASE / args.source))
    model = YOLO(ckpt)
    model.predict(source=src, imgsz=args.imgsz, conf=args.conf, show=False, save=True,
                  project=str(BASE / "experiments"), name="predict", exist_ok=True)


if __name__ == "__main__":
    main()
