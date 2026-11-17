"""Stage 2 - Train a RandomForest activity classifier on the collected keypoints.

Example:
    python train_model.py --input dataset.csv --output activity_model.pkl
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pose_utils import FEATURE_COLUMNS, LABEL_COLUMN, NUM_FEATURES, NUM_KEYPOINTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the activity classifier.")
    p.add_argument("--input", default="dataset.csv", help="CSV from collect_data.py.")
    p.add_argument("--output", default="activity_model.pkl", help="Where to save the model.")
    p.add_argument("--max-missing", type=int, default=8,
                   help="Drop rows with more than this many undetected keypoints (of 17). Default 8.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default 0.2).")
    p.add_argument("--trees", type=int, default=200, help="Number of trees (default 200).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_and_clean(path: str, max_missing: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the CSV and return (X, y) with only usable rows.

    Rows are dropped if they have NaNs / no label, a wrong feature count, or too
    many undetected keypoints. Remaining zeros (a few occluded joints) are kept
    as-is on purpose: live inference feeds the model the same zeros, so training
    on imputed values would create a train/inference mismatch.
    """
    df = pd.read_csv(path)
    missing_cols = [c for c in [LABEL_COLUMN, *FEATURE_COLUMNS] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"{path} is missing expected columns: {missing_cols[:4]}...")

    n_start = len(df)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS)

    feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    assert feats.shape[1] == NUM_FEATURES
    pairs = feats.reshape(-1, NUM_KEYPOINTS, 2)
    missing = np.all(pairs == 0.0, axis=2).sum(axis=1)  # undetected keypoints per row
    keep = missing <= max_missing
    df = df[keep]

    print(f"Loaded {n_start} rows; kept {len(df)} after cleaning "
          f"(dropped {n_start - len(df)} with NaNs or >{max_missing} missing keypoints).")
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32), df[LABEL_COLUMN].astype(str).to_numpy()


def main() -> None:
    args = parse_args()
    X, y = load_and_clean(args.input, args.max_missing)

    labels, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(labels, counts.tolist())))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 classes (e.g. 'idle' and an action). Collect more data first.")
    if counts.min() < 2:
        raise SystemExit(f"Class {labels[counts.argmin()]!r} has <2 samples; collect more data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.trees, random_state=args.seed, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}\n")
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))

    joblib.dump(clf, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
