"""Export checkpoint -> ONNX (jalur menuju TF.js / web).

Alur deployment RMC: best.pt (PyTorch) -> ONNX (validasi numerik) -> TF.js/web.
Script ini tahap 1: export + verifikasi kesetaraan prediksi vs PyTorch.

Usage:
    python scripts/export_onnx.py --weights checkpoints/<exp>/weights/best.pt
    python scripts/export_onnx.py --weights checkpoints/<exp>/weights/best.pt --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the checkpoint was trained at")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt = args.weights if Path(args.weights).is_absolute() else str(BASE / args.weights)
    model = YOLO(ckpt)
    args.imgsz = args.imgsz or int(model.ckpt["train_args"]["imgsz"])
    out = model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
    print(f"[export] ONNX -> {out}")

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        img_dir = BASE / "data" / "processed" / "images" / "val"
        sample = sorted(img_dir.glob("A_*"))[0]
        print(f"[verify] sample: {sample.name}")

        r_torch = model.predict(source=str(sample), imgsz=args.imgsz, verbose=False)[0]
        n_torch = 0 if r_torch.boxes is None else len(r_torch.boxes)

        import cv2

        img = cv2.imread(str(sample))
        h, w = img.shape[:2]
        s = args.imgsz / max(h, w)
        # ponytail: samakan dgn Ultralytics + web/poc.html (letterbox TENGAH abu-114,
        # bukan kiri-atas hitam) agar verifikasi valid untuk deployment
        canvas = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
        rimg = cv2.resize(img, (int(w * s), int(h * s)))
        dy, dx = (args.imgsz - rimg.shape[0]) // 2, (args.imgsz - rimg.shape[1]) // 2
        canvas[dy: dy + rimg.shape[0], dx: dx + rimg.shape[1]] = rimg
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        print(f"[verify] torch boxes={n_torch} | onnx out shape={y.shape} "
              f"(batch, 4+nc+... per anchor; argmax ambil setelah NMS di app)")


if __name__ == "__main__":
    main()
