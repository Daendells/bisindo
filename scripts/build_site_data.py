"""Build the website's data: docs/results.json, docs/samples/*.jpg, docs/model/*.onnx.

Every number on the results page is computed here from the prediction files that
evaluate.py / evaluate_holdout.py wrote, so the page cannot drift from the runs.
Top-1 protocol: one prediction per image (highest-confidence box); no box = wrong.

Usage:
    python scripts/build_site_data.py --model 20260911_225041
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
EXP = BASE / "experiments"
DOCS = BASE / "docs"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONF_MIN = 0.6  # the web app's "not sure" threshold; reported against val below
EXTERNAL = BASE / "data" / "holdout_achmadnoer" / "Citra BISINDO"
SAMPLES = [("wall white (1).jpg", "plain wall"), ("body dot (1).jpg", "dotted shirt"),
           ("body white (1).jpg", "white shirt")]

# YOLO runs shown in the comparison table, in display order: experiment id -> label
YOLO_RUNS = {
    "20261020_130346": "YOLO11n · 416 px",
    "20260911_225041": "YOLO11n · 320 px",
    "20260916_112443": "YOLOv8n · 320 px",
}


def eval_items(csv_path: Path) -> list:
    """evaluate.py rows (several per image); the truth is the file-name prefix."""
    per = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            per[r["image"]].append(r)
    items = []
    for img, rows in sorted(per.items()):
        preds = [r for r in rows if r["flag"] == "pred"]
        best = max(preds, key=lambda r: float(r["conf"])) if preds else None
        items.append((img.split("_")[0],
                      LETTERS[int(best["pred_cls"])] if best else None,
                      float(best["conf"]) if best else 0.0))
    return items


def external_items(csv_path: Path) -> list:
    with csv_path.open() as f:
        return [(r["true"], r["pred"] or None, float(r["conf"] or 0)) for r in csv.DictReader(f)]


def summarize(items: list) -> dict:
    n = len(items)
    correct = sum(t == p for t, p, _ in items)
    per, f1s = {}, []
    for L in LETTERS:
        tp = sum(t == L and p == L for t, p, _ in items)
        fp = sum(p == L and t != L for t, p, _ in items)
        fn = sum(t == L and p != L for t, p, _ in items)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        wrong = Counter(p or "no detection" for t, p, _ in items if t == L and p != L)
        per[L] = {"n": tp + fn, "ok": tp, "confused": dict(wrong.most_common())}
    assert sum(v["n"] for v in per.values()) == n, "an image has a label outside A-Z"
    return {"n": n, "correct": correct, "top1": round(correct / n, 6),
            "macro_f1": round(float(np.mean(f1s)), 4), "per_letter": per}


def threshold_stats(items: list) -> dict:
    kept = [(t, p) for t, p, c in items if c >= CONF_MIN]
    return {"conf": CONF_MIN, "coverage": round(len(kept) / len(items), 6),
            "precision": round(sum(t == p for t, p in kept) / len(kept), 6)}


def onnx_for(exp_id: str) -> Path:
    from ultralytics import YOLO

    pt = BASE / "checkpoints" / exp_id / "weights" / "best.pt"
    onnx = pt.with_suffix(".onnx")
    if not onnx.exists():
        model = YOLO(str(pt))
        model.export(format="onnx", imgsz=int(model.ckpt["train_args"]["imgsz"]), dynamic=False)
    return onnx


def cpu_latency_ms(onnx: Path, runs: int = 50) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.default_rng(0).random(inp.shape, dtype=np.float32)
    for _ in range(5):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def yolo_facts(exp_id: str) -> dict:
    import torch

    ckpt = torch.load(BASE / "checkpoints" / exp_id / "weights" / "best.pt",
                      map_location="cpu", weights_only=False)
    onnx = onnx_for(exp_id)
    return {"imgsz": int(ckpt["train_args"]["imgsz"]),
            "params_m": round(sum(p.numel() for p in ckpt["model"].parameters()) / 1e6, 1),
            "size_mb": round(onnx.stat().st_size / 1e6, 1),
            "cpu_ms": cpu_latency_ms(onnx)}


def audit() -> dict:
    v1 = json.loads((BASE / "data/reports/split_report.json").read_text())
    v2 = json.loads((BASE / "data/reports/split_report_v2.json").read_text())
    near = json.loads((BASE / "data/reports/near_dup_audit_v1.json").read_text())  # original split
    near_after = json.loads((BASE / "data/reports/near_dup_audit.json").read_text())  # group split
    with (EXP / "runs.csv").open() as f:
        runs = {r["experiment_id"]: r for r in csv.DictReader(f)}
    return {
        "raw_images": v1["raw_train"] + v1["raw_val"],
        "exact_duplicates": v1["train_within_dupes_dropped"] + v1["val_within_dupes_dropped"],
        "cross_split_copies": v1["cross_split_leaked_dropped_from_val"],
        "val_near_copy_share": near["frac_le_6"],
        "val_near_copy_share_after": near_after["frac_le_6"],
        "phash_groups": v2["groups"],
        "split": v2["final"],
        "leaky_split_map50_95": float(runs["20260911_205035"]["val_map50_95"]),
        "clean_split_map50_95": float(runs["20260912_012657"]["val_map50_95"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="experiment id of the deployed model")
    args = ap.parse_args()
    runs = dict(YOLO_RUNS)
    if args.model not in runs:
        runs[args.model] = "YOLO11n · {} px"  # imgsz filled in below

    comparison, deployed = [], None
    for exp_id, label in runs.items():
        facts = yolo_facts(exp_id)
        val = summarize(eval_items(EXP / f"{exp_id}_val_eval" / "predictions.csv"))
        ext = summarize(external_items(EXP / f"{exp_id}_external_eval" / "predictions.csv"))
        row = {"id": exp_id, "model": label.format(facts["imgsz"]), "approach": "Detector, fine-tuned end to end",
               "val_top1": val["top1"], "external_top1": ext["top1"], **facts,
               "val_correct": val["correct"], "val_n": val["n"],
               "external_correct": ext["correct"], "external_n": ext["n"],
               "val_mn": val["per_letter"]["M"]["confused"].get("N", 0) + val["per_letter"]["N"]["confused"].get("M", 0),
               "deployed": exp_id == args.model}
        comparison.append(row)
        if row["deployed"]:
            deployed = (row, val, ext)

    svm = json.loads((EXP / "svm_final_eval/summary.json").read_text())
    comparison.append({"model": "Hand landmarks + SVM", "approach": "MediaPipe Hands (one hand) + RBF SVM",
                       "val_top1": svm["val"]["top1_global"], "external_top1": svm["holdout"]["top1_global"],
                       "deployed": False})
    dino = json.loads((EXP / "dino_probe/summary.json").read_text())
    comparison.append({"model": "DINOv2 ViT-S/14 + SVM", "approach": "Frozen image embedding + RBF SVM",
                       "val_top1": dino["val"], "external_top1": dino["holdout"], "deployed": False})

    row, val_sum, ext_sum = deployed
    test_csv = EXP / f"{args.model}_test_eval" / "predictions.csv"
    results = {
        "model": {**{k: row[k] for k in ("id", "model", "imgsz", "params_m", "size_mb", "cpu_ms")},
                  "file": "model/bisindo-yolo11n.onnx"},
        "eval": {"val": val_sum, "external": ext_sum,
                 "test": summarize(eval_items(test_csv)) if test_csv.exists() else None},
        "threshold": threshold_stats(eval_items(EXP / f"{args.model}_val_eval" / "predictions.csv")),
        "audit": audit(),
        "comparison": comparison,
        "samples": [],
    }

    (DOCS / "model").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_for(args.model), DOCS / results["model"]["file"])

    import cv2

    out = DOCS / "samples"
    out.mkdir(exist_ok=True)
    for L in LETTERS:
        for i, (name, background) in enumerate(SAMPLES, 1):
            img = cv2.imread(str(EXTERNAL / L / name))
            s = 480 / max(img.shape[:2])
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            file = f"samples/{L}-{i}.jpg"
            cv2.imwrite(str(DOCS / file), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            results["samples"].append({"file": file, "letter": L, "background": background})

    # Hero figure: the deployed model's real output on one unseen-signer sample.
    from ultralytics import YOLO

    hero = "samples/B-1.jpg"
    r = YOLO(str(BASE / "checkpoints" / args.model / "weights" / "best.pt")).predict(
        str(DOCS / hero), imgsz=row["imgsz"], device="cpu", verbose=False)[0]
    b = max(r.boxes, key=lambda b: float(b.conf))
    results["hero"] = {"file": hero, "letter": LETTERS[int(b.cls)], "conf": round(float(b.conf), 3),
                       "box": [round(float(v), 4) for v in b.xyxyn[0].tolist()]}

    (DOCS / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results["eval"][k] and {kk: results["eval"][k][kk] for kk in ("n", "correct", "top1", "macro_f1")}
                      for k in results["eval"]}, indent=1))
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
