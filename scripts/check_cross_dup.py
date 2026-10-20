"""Cross-split duplicate check."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

def hashes(split):
    d = {}
    for p in (ROOT / "images" / split).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg"):
            d[hashlib.md5(p.read_bytes()).hexdigest()] = str(p)
    return d

train = hashes("train")
val = hashes("val")
cross = set(train) & set(val)
print(f"train files: {len(train)} unique hashes")
print(f"val files: {len(val)} unique hashes")
print(f"CROSS-SPLIT duplicates (same md5 in train AND val): {len(cross)}")
for h in list(cross)[:10]:
    print(f"  train: {train[h]}")
    print(f"  val:   {val[h]}")
