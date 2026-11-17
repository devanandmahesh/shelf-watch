"""Train the time-window classifier and measure it honestly (whole videos / scenes held out).

Reads one or more CSVs from kaggle_to_windows.py / folder_to_windows.py. First it cross-validates
with whole groups held out (a group is a Kaggle scene, or one video from folder_to_windows.py),
judging both single windows and per-video verdicts (alert when "conceal" is predicted for
--alert-frames windows in a row, the same rule demo.py uses). With --test it also judges a dataset
the model never trained on. Then it trains the final model on ALL --input data and saves it.

Examples:
    python3 train_window_model.py                                        # Kaggle only
    python3 train_window_model.py --input archive_windows.csv --test kaggle_windows.csv
    python3 train_window_model.py --input archive_windows.csv kaggle_windows.csv \\
        --output shoplift_window_model.pkl
    python3 demo.py --model shoplift_window_model.pkl
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from window_utils import N_STEPS, WINDOW_COLUMNS, WINDOW_S


def load(paths: list[str]) -> pd.DataFrame:
    """Concatenate window CSVs, giving every row a group (the unit held out when testing)."""
    parts = []
    for path in paths:
        df = pd.read_csv(path).copy()  # copy() de-fragments the 340-column frame
        if "group" not in df.columns:  # kaggle_windows.csv: a scene = its two clips, same room
            df["group"] = "scene" + df["clip"].astype(str).str.extract(r"(\d+)$")[0]
        df["group"] = os.path.basename(path) + ":" + df["group"].astype(str)
        df["clip"] = os.path.basename(path) + ":" + df["clip"].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def verdict(pred: np.ndarray, frames: np.ndarray, need: int) -> bool:
    """True if 'conceal' is predicted for `need` consecutive windows of one video."""
    step = int(np.diff(frames).min()) if len(frames) > 1 else 1  # 1, or the frame stride
    streak, prev = 0, None
    for f, label in zip(frames, pred):
        if label == "conceal":
            streak = streak + 1 if (prev is not None and f - prev == step) else 1
        else:
            streak = 0
        prev = f
        if streak >= need:
            return True
    return False


def evaluate(clf: RandomForestClassifier, df: pd.DataFrame, X: np.ndarray, need: int) -> dict[str, object]:
    """Window and per-video scores of ``clf`` on ``df``."""
    y = df["label"].astype(str).to_numpy()
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, list(clf.classes_).index("conceal")]
    r: dict[str, object] = {
        "tp": int(((pred == "conceal") & (y == "conceal")).sum()), "pos": int((y == "conceal").sum()),
        "fp": int(((pred == "conceal") & (y == "normal")).sum()), "neg": int((y == "normal").sum()),
        "y": y, "proba": proba, "caught": 0, "thefts": 0, "flagged": 0, "normals": 0,
    }
    clips = df["clip"].to_numpy()
    frames = df["frame"].to_numpy()
    for c in pd.unique(clips):
        m = clips == c
        truth = bool((y[m] == "conceal").any())
        got = verdict(pred[m], frames[m], need)
        if truth:
            r["thefts"], r["caught"] = r["thefts"] + 1, r["caught"] + got  # type: ignore[operator]
        else:
            r["normals"], r["flagged"] = r["normals"] + 1, r["flagged"] + got  # type: ignore[operator]
    return r


def summary(r: dict[str, object]) -> str:
    return (f"windows: caught {r['tp']}/{r['pos']}, false {r['fp']}/{r['neg']} | "
            f"videos: shoplifting caught {r['caught']}/{r['thefts']}, normal falsely flagged {r['flagged']}/{r['normals']}")


def auc(y: np.ndarray, proba: np.ndarray) -> str:
    return f"{roc_auc_score(y == 'conceal', proba):.2f}" if len(set(y)) == 2 else "n/a"


def forest(args: argparse.Namespace) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=-1,
                                  class_weight="balanced")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the time-window classifier.")
    p.add_argument("--input", nargs="+", default=["kaggle_windows.csv"], help="Training CSV(s).")
    p.add_argument("--test", nargs="*", default=[], help="CSV(s) never trained on, judged at the end.")
    p.add_argument("--output", default="kaggle_window_model.pkl")
    p.add_argument("--folds", type=int, default=5, help="Cross-validation folds (fewer if fewer groups).")
    p.add_argument("--alert-frames", type=int, default=8, help="Consecutive windows for a video verdict.")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load(args.input)
    X = df[WINDOW_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    groups = df["group"].to_numpy()
    n_groups = len(set(groups))
    print(f"Loaded {len(df)} windows from {df['clip'].nunique()} videos, {n_groups} groups: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    folds = min(args.folds, n_groups)
    print(f"\nCROSS-VALIDATION ({folds} folds, whole groups held out)")
    totals = {"tp": 0, "pos": 0, "fp": 0, "neg": 0, "caught": 0, "thefts": 0, "flagged": 0, "normals": 0}
    all_y, all_p = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups), 1):
        clf = forest(args).fit(X[tr], y[tr])
        r = evaluate(clf, df.iloc[te], X[te], args.alert_frames)
        print(f"  fold {k}: {summary(r)}")
        for key in totals:
            totals[key] += r[key]  # type: ignore[operator]
        all_y.append(r["y"])
        all_p.append(r["proba"])
    print(f"  => {summary(totals)} | AUC {auc(np.concatenate(all_y), np.concatenate(all_p))}")

    clf = forest(args).fit(X, y)
    for path in args.test:
        tdf = load([path])
        r = evaluate(clf, tdf, tdf[WINDOW_COLUMNS].to_numpy(dtype=np.float32), args.alert_frames)
        print(f"\nUNSEEN DATASET {path}: {summary(r)} | AUC {auc(r['y'], r['proba'])}")  # type: ignore[arg-type]

    joblib.dump({"kind": "window", "model": clf, "window_s": WINDOW_S, "n_steps": N_STEPS}, args.output)
    print(f"\nFinal model (trained on all --input data) saved to {args.output}")


if __name__ == "__main__":
    main()
