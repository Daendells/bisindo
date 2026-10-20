"""Baseline: hand landmarks + RBF SVM (run extract_landmarks.py first, with --feat xyz and --feat geo).

Grid: C {0.01..1000} x gamma {scale, auto, 0.1..0.0001} x features {xyz 63-D, geo 30-D, concat 93-D},
scored by macro-F1 with stratified 5-fold CV on the training split. The best CV setting is then
scored once on val and on the unseen-signer set; a photo where MediaPipe finds no hand counts as wrong.

Usage: python scripts/tune_svm_final.py
Writes: experiments/svm_final_eval/{summary.json,confusion_val.csv,confusion_holdout.csv,svm.joblib}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib

    feats = {}
    for f in ["xyz", "geo"]:
        tr = np.load(BASE / "experiments/landmarks/train_geo.npz"
                     if f == "geo" else BASE / "experiments/landmarks/train.npz")
        feats[f] = (tr["X"], tr["y"])
    Xc = np.concatenate([feats["xyz"][0], feats["geo"][0]], axis=1)
    feats["concat"] = (Xc, feats["xyz"][1])

    results = {}
    best = (-1, None, None)
    for f, (X, y) in feats.items():
        gs = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
            {"svc__C": [0.01, 0.1, 1, 10, 100, 1000],
             "svc__gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]},
            scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
        t0 = time.perf_counter()
        gs.fit(X, y)
        dt = round(time.perf_counter() - t0)
        results[f] = {"best": gs.best_params_, "cv_f1": round(float(gs.best_score_), 4),
                      "secs": dt}
        print(f, results[f], flush=True)
        if gs.best_score_ > best[0]:
            best = (gs.best_score_, f, gs.best_estimator_)

    name, clf = best[1], best[2]
    print("WINNER:", name)
    out = BASE / "experiments/svm_final_eval"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "svm.joblib")

    rep = {"feat": name, "best_params": results[name]["best"],
           "cv_f1_macro": results[name]["cv_f1"], "all": results}
    for split, n_all in [("val", 1194), ("holdout", 312)]:
        fx = f"_{name}" if name != "xyz" else ""
        if name == "concat":
            a = np.load(BASE / "experiments/landmarks" / f"{split}.npz")
            b = np.load(BASE / "experiments/landmarks" / f"{split}_geo.npz")
            X = np.concatenate([a["X"], b["X"]], axis=1)
            y = a["y"]
        else:
            d = np.load(BASE / "experiments/landmarks" / f"{split}{fx}.npz")
            X, y = d["X"], d["y"]
        pred = clf.predict(X) if len(X) else np.array([], dtype=int)
        ok = int((pred == y).sum())
        cm = np.zeros((26, 26), dtype=int)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
        np.savetxt(out / f"confusion_{split}.csv", cm, delimiter=",", fmt="%d")
        rep[split] = {"n": n_all, "n_detected": len(X),
                      "top1_global": round(ok / n_all, 4)}
    (out / "summary.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "all"}, indent=2))


if __name__ == "__main__":
    main()
