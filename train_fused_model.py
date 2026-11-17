"""Train ONE model that fuses the pose-window stream and the video stream, and measure it honestly.

Reads video_embeddings.npz (videos_to_embeddings.py) and the window CSVs (folder_to_windows.py /
kaggle_to_windows.py). Each stream keeps its own classifier — a logistic regression on the frozen
MViT-V2-S clip embeddings, a random forest on the 340 window features — and a person's score is the
average of the two streams' 90th percentiles. One threshold covers the pair, so demo.py has one
model file and one alert rule instead of three.

Tests:
  1. Cross-validation on the archive with whole videos held out, reporting each stream alone and
     the fusion, so you can see what fusing is actually worth.
  2. Both streams trained on the archive only, judged on the Kaggle videos (other camera, room).
The threshold is the one that falsely flags at most --max-false of the held-out archive normals.

Example:
    python3 train_fused_model.py
    python3 demo.py --model fused_model.pkl
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_video_model import VIDEO_QUANTILE, classifier
from video_utils import CLIP_FRAMES, CLIP_S
from window_utils import N_STEPS, WINDOW_COLUMNS, WINDOW_S


def forest(trees: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1,
                                  class_weight="balanced")


def auc(y: np.ndarray, s: np.ndarray) -> str:
    return f"{roc_auc_score(y, s):.3f}" if len(set(y.tolist())) == 2 else "n/a"


def windows(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, is_conceal, video) from one or more window CSVs."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return (df[WINDOW_COLUMNS].to_numpy(np.float32), (df["label"] == "conceal").to_numpy(),
            df["clip"].to_numpy().astype(str))


def main() -> None:
    p = argparse.ArgumentParser(description="Train the fused pose + video model.")
    p.add_argument("--embeddings", default="video_embeddings.npz")
    p.add_argument("--archive-windows", default="archive_windows.csv")
    p.add_argument("--kaggle-windows", default="kaggle_windows.csv")
    p.add_argument("--output", default="fused_model.pkl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--c", type=float, default=0.01, help="Video head's logistic regression C.")
    p.add_argument("--trees", type=int, default=200, help="Pose head's random forest size.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-false", type=float, default=0.1,
                   help="Pick the threshold that flags at most this share of normal videos.")
    args = p.parse_args()

    d = np.load(args.embeddings)
    ds = d["dataset"].astype(str)
    emb, vlab, vclip = d["emb"], d["label"].astype(str) == "conceal", d["clip"].astype(str)
    arch, kag = ds == "archive", ds == "kaggle"
    Xw, yw, wclip = windows([args.archive_windows])

    videos = sorted(set(vclip[arch]) & set(wclip))
    theft = np.array([bool(vlab[vclip == v].any()) for v in videos])
    print(f"{len(videos)} archive videos both streams can score ({int(theft.sum())} shoplifting); "
          f"{arch.sum()} clip embeddings, {len(Xw)} pose windows")

    def per_video(pv: np.ndarray, vmask: np.ndarray, pw: np.ndarray, wmask: np.ndarray,
                  names: list[str], vc: np.ndarray, wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each stream's VIDEO_QUANTILE per video, from that stream's clip/window probabilities."""
        return (np.array([np.quantile(pv[vc[vmask] == v], VIDEO_QUANTILE) for v in names]),
                np.array([np.quantile(pw[wc[wmask] == v], VIDEO_QUANTILE) for v in names]))

    vid = np.zeros(len(videos))
    pose = np.zeros(len(videos))
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr_i, te_i in folds.split(videos, theft):
        tr = [videos[i] for i in tr_i]
        te = [videos[i] for i in te_i]
        vm_tr, vm_te = arch & np.isin(vclip, tr), arch & np.isin(vclip, te)
        wm_tr, wm_te = np.isin(wclip, tr), np.isin(wclip, te)
        pv = classifier(args.c).fit(emb[vm_tr], vlab[vm_tr]).predict_proba(emb[vm_te])[:, 1]
        pw = forest(args.trees, args.seed).fit(Xw[wm_tr], yw[wm_tr]).predict_proba(Xw[wm_te])[:, 1]
        v, q = per_video(pv, vm_te, pw, wm_te, te, vclip, wclip)
        vid[te_i], pose[te_i] = v, q
        print(f"  fold done: {len(te)} videos held out", flush=True)

    fused = (vid + pose) / 2

    def own_threshold(s: np.ndarray, y: np.ndarray) -> float:
        """The threshold that spends exactly the --max-false budget on these normal videos."""
        return float(np.quantile(s[~y], 1 - args.max_false))

    def at_own_threshold(s: np.ndarray, y: np.ndarray) -> str:
        """Each stream judged at ITS OWN threshold for the same false-alarm budget, so the
        streams are comparable: a lower threshold is not a merit."""
        t = own_threshold(s, y)
        return (f"thr {t:.2f}  caught {int((s[y] > t).sum()):>2d}/{int(y.sum())}  "
                f"falsely flagged {int((s[~y] > t).sum()):>2d}/{int((~y).sum())}")

    thr, vthr = own_threshold(fused, theft), own_threshold(vid, theft)
    print(f"\nARCHIVE, whole videos held out ({args.folds} folds)")
    for name, s in (("pose window only", pose), ("video only", vid), ("FUSED", fused)):
        print(f"  {name:17s} video AUC {auc(theft, s)}   {at_own_threshold(s, theft)}")
    print(f"  the two streams' video scores correlate {np.corrcoef(pose, vid)[0, 1]:.2f}")
    print(f"  saved thresholds: fused {thr:.2f}, and {vthr:.2f} for the video stream alone, which "
          f"demo.py falls back to when a person's keypoints drop out")

    if kag.any():
        Xk, yk, kclip = windows([args.kaggle_windows])
        kvideos = sorted(set(vclip[kag]) & set(kclip))
        ktheft = np.array([bool(vlab[vclip == v].any()) for v in kvideos])
        pv = classifier(args.c).fit(emb[arch], vlab[arch]).predict_proba(emb[kag])[:, 1]
        pw = forest(args.trees, args.seed).fit(Xw, yw).predict_proba(Xk)[:, 1]
        kv, kp = per_video(pv, kag, pw, np.ones(len(Xk), bool), kvideos, vclip, kclip)
        kf = (kv + kp) / 2
        print(f"\nKAGGLE, never trained on ({len(kvideos)} videos, so weak evidence)")
        for name, s in (("pose window only", kp), ("video only", kv), ("FUSED", kf)):
            print(f"  {name:17s} video AUC {auc(ktheft, s)}   {at_own_threshold(s, ktheft)}")
        print(f"  fused threshold {thr:.2f}: shoplifting caught {int((kf[ktheft] > thr).sum())}/"
              f"{int(ktheft.sum())}, normal falsely flagged {int((kf[~ktheft] > thr).sum())}/"
              f"{int((~ktheft).sum())}")
        if roc_auc_score(ktheft, kf) < 0.7:
            print("  WARNING: fusing does not fix the camera change — on a camera it was not trained "
                  "on this model is still near chance, and the threshold above does not carry over. "
                  "Calibrate on footage from your own camera: demo.py --calibrate.")

    vhead = classifier(args.c).fit(emb, vlab)
    phead = forest(args.trees, args.seed).fit(*windows([args.archive_windows, args.kaggle_windows])[:2])
    joblib.dump({"kind": "fused", "threshold": thr, "quantile": VIDEO_QUANTILE,
                 "video": {"kind": "video", "model": vhead, "threshold": vthr,
                           "quantile": VIDEO_QUANTILE, "clip_s": CLIP_S, "clip_frames": CLIP_FRAMES,
                           "backbone": "mvit_v2_s/KINETICS400_V1"},
                 "pose": {"model": phead, "window_s": WINDOW_S, "n_steps": N_STEPS}}, args.output)
    print(f"\nFinal fused model (archive + Kaggle, both streams) saved to {args.output}")


if __name__ == "__main__":
    main()
