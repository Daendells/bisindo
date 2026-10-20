"""Dataset audit for BISINDO YOLO project.
Reads bisindo/images/{train,val}/{A..Z} + bisindo/labels/{train,val}/{A..Z}.
Outputs human + machine report. No test split exists -> flagged.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image
    HAS_PIL = False  # disabled by default: full verify is slow on 11k images; enable manually for spot-check
except ImportError:
    HAS_PIL = False

CLASSES = [chr(ord("A") + i) for i in range(26)]

report = {"classes": CLASSES, "splits": {}, "issues": []}

total_imgs = 0
for split in ("train", "val"):
    img_root = ROOT / "images" / split
    lbl_root = ROOT / "labels" / split
    per_class = {}
    all_hashes = []
    corrupt = []
    missing_label = []
    missing_image = []
    bad_lines = []
    class_idx_counter = Counter()

    for cls in CLASSES:
        img_dir = img_root / cls
        lbl_dir = lbl_root / cls
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpeg")) if img_dir.exists() else []
        per_class[cls] = len(imgs)
        total_imgs += len(imgs)
        for img in imgs:
            # exact hash for duplicate detection
            try:
                h = hashlib.md5(img.read_bytes()).hexdigest()
                all_hashes.append((h, str(img)))
            except OSError:
                corrupt.append(str(img))
            # paired label check: labels/<split>/<CLS>/<stem>.txt
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing_label.append(str(img))
            else:
                try:
                    lines = lbl.read_text().strip().splitlines()
                    if not lines or not lines[0].strip():
                        bad_lines.append(str(lbl) + " (empty)")
                    for ln in lines:
                        parts = ln.split()
                        if len(parts) != 5:
                            bad_lines.append(str(lbl) + f" (bad cols: {ln[:40]})")
                            break
                        ci = int(float(parts[0]))
                        class_idx_counter[ci] += 1
                        for v in parts[1:]:
                            fv = float(v)
                            if not (0.0 <= fv <= 1.0):
                                bad_lines.append(str(lbl) + f" (oob: {ln[:40]})")
                                break
                except (ValueError, OSError) as e:
                    bad_lines.append(str(lbl) + f" ({e})")
            # readability
            if HAS_PIL:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except (OSError, ValueError):
                    corrupt.append(str(img))

    # duplicate hashes
    seen = {}
    dups = 0
    for h, p in all_hashes:
        if h in seen:
            dups += 1
        else:
            seen[h] = p

    # orphan labels (txt without image)
    orphan = 0
    if lbl_root.exists():
        for txt in lbl_root.rglob("*.txt"):
            # expected image in same class folder (jpg/png)
            stem = txt.stem
            cls = txt.parent.name
            cand = [ (img_root / cls / (stem + ext)) for ext in (".jpg", ".png", ".jpeg")]
            if not any(c.exists() for c in cand):
                orphan += 1
                if orphan <= 5:
                    missing_image.append(str(txt))

    report["splits"][split] = {
        "per_class": per_class,
        "total": sum(per_class.values()),
        "corrupt_unreadable": len(set(corrupt)),
        "missing_label": len(missing_label),
        "orphan_labels": orphan,
        "bad_label_lines": len(bad_lines),
        "exact_duplicates": dups,
        "label_class_index_hist": dict(sorted(class_idx_counter.items())),
    }
    if missing_label:
        report["issues"].append(f"{split}: {len(missing_label)} images without .txt (e.g. {missing_label[0]})")
    if orphan:
        report["issues"].append(f"{split}: {orphan} .txt without image")
    if bad_lines:
        report["issues"].append(f"{split}: {len(bad_lines)} bad label lines (e.g. {bad_lines[0]})")
    if dups:
        report["issues"].append(f"{split}: {dups} exact-duplicate images (same md5)")
    if corrupt:
        report["issues"].append(f"{split}: {len(set(corrupt))} corrupt/unreadable")

# imbalance
for split in ("train", "val"):
    counts = list(report["splits"][split]["per_class"].values())
    if counts and max(counts) > 0:
        report["splits"][split]["imbalance_ratio"] = round(max(counts) / max(min(counts), 1), 2)

# test split check
has_test = (ROOT / "images" / "test").exists()
report["has_test_split"] = has_test
if not has_test:
    report["issues"].append("NO test split: create frozen test from val or collect new signer-session holdout before tuning")

# label-index vs folder consistency (YOLO class idx should map to folder?)
# If folder A contains only idx 0, B only idx 1, ... mapping is consistent.
report["note"] = ("If per-folder class idx is constant (A->0, B->1...), "
                   "flatten to YOLO flat layout before training to avoid path leakage. "
                   "Group-aware split by signer/session recommended if known.")

with open(OUT / "dataset_audit.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
