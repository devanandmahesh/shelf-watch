"""Train the classifier on top of the video-model embeddings and measure it honestly.

Reads video_embeddings.npz (videos_to_embeddings.py). The pretrained MViT-V2-S stays frozen; only a
logistic regression on its 768-number clip embeddings is trained, which suits ~180 videos.

Tests:
  1. Cross-validation on the archive with whole videos held out.
  2. Trained on the archive only, judged on the Kaggle videos (other camera, other room).
A video's score is the 90th percentile of its clip probabilities (its most suspicious moments), so
a normal video needs more than one odd clip to look suspicious. The alert threshold printed and saved
is the one that falsely flags at most --max-false of the held-out archive normal videos.

Test 2 is the one that says whether the model is usable anywhere else: a threshold fitted on one
camera's normal footage means nothing on a camera whose scores sit higher, so the Kaggle lines are
checked against the saved threshold and warned about below.

The final model is trained on archive + Kaggle and saved for demo.py.

Example:
    python3 train_video_model.py
    python3 demo.py --model video_model.pkl
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from video_utils import CLIP_FRAMES, CLIP_S

VIDEO_QUANTILE = 0.9


def classifier(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000))


def video_scores(p: np.ndarray, clips: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(score, is_theft) per video: score = VIDEO_QUANTILE of its clip probabilities."""
    df = pd.DataFrame({"p": p, "clip": clips, "theft": labels == "conceal"})
    g = df.groupby("clip")
    return g["p"].quantile(VIDEO_QUANTILE).to_numpy(), g["theft"].any().to_numpy()


def auc(y: np.ndarray, s: np.ndarray) -> str:
    return f"{roc_auc_score(y, s):.2f}" if len(set(y.tolist())) == 2 else "n/a"


def main() -> None:
    p = argparse.ArgumentParser(description="Train the video-model classifier.")
    p.add_argument("--input", default="video_embeddings.npz")
    p.add_argument("--output", default="video_model.pkl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--c", type=float, default=0.01, help="Logistic regression C (smaller = simpler).")
    p.add_argument("--max-false", type=float, default=0.1,
                   help="Pick the threshold that flags at most this share of normal videos (default 0.1).")
    args = p.parse_args()

    d = np.load(args.input)
    X, y, clips, groups, dataset = d["emb"], d["label"].astype(str), d["clip"].astype(str), d["group"].astype(str), d["dataset"].astype(str)
    arch, kag = dataset == "archive", dataset == "kaggle"
    print(f"Loaded {len(X)} clips: archive {arch.sum()} from {len(set(clips[arch]))} videos, "
          f"Kaggle {kag.sum()} from {len(set(clips[kag]))} videos")

    Xa, ya, ca, ga = X[arch], y[arch], clips[arch], groups[arch]
    oof = np.zeros(len(Xa))
    for tr, te in GroupKFold(n_splits=args.folds).split(Xa, ya, ga):
        oof[te] = classifier(args.c).fit(Xa[tr], ya[tr] == "conceal").predict_proba(Xa[te])[:, 1]
    vs, vt = video_scores(oof, ca, ya)
    thr = float(np.quantile(vs[~vt], 1 - args.max_false))
    print(f"\nARCHIVE, whole videos held out ({args.folds} folds)")
    print(f"  clip AUC {auc(ya == 'conceal', oof)} | video AUC {auc(vt, vs)}")
    print(f"  threshold {thr:.2f}: shoplifting videos caught {int((vs[vt] > thr).sum())}/{int(vt.sum())}, "
          f"normal falsely flagged {int((vs[~vt] > thr).sum())}/{int((~vt).sum())}")

    if kag.any():
        pk = classifier(args.c).fit(Xa, ya == "conceal").predict_proba(X[kag])[:, 1]
        ks, kt = video_scores(pk, clips[kag], y[kag])
        print(f"\nKAGGLE (never trained on): clip AUC {auc(y[kag] == 'conceal', pk)} | video AUC {auc(kt, ks)}")
        print(f"  threshold {thr:.2f}: shoplifting videos caught {int((ks[kt] > thr).sum())}/{int(kt.sum())}, "
              f"normal falsely flagged {int((ks[~kt] > thr).sum())}/{int((~kt).sum())}")

        if roc_auc_score(kt, ks) < 0.7 or (ks[~kt] > thr).mean() > args.max_false:
            print("  WARNING: on a camera it was not trained on this model barely separates theft "
                  "from normal and its scores run high, so the threshold above does not carry over. "
                  "Before trusting alerts on another camera, record normal footage from THAT camera "
                  "and raise demo.py --threshold until it stops firing on it.")

    final = classifier(args.c).fit(X, y == "conceal")  # predict_proba[:, 1] = P(conceal)
    joblib.dump({"kind": "video", "model": final, "threshold": thr, "quantile": VIDEO_QUANTILE,
                 "clip_s": CLIP_S, "clip_frames": CLIP_FRAMES, "backbone": "mvit_v2_s/KINETICS400_V1"}, args.output)
    print(f"\nFinal model (archive + Kaggle) saved to {args.output}")


if __name__ == "__main__":
    main()
