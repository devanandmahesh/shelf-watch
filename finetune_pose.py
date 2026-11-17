"""Fine-tune yolo11n-pose on the Kaggle CCTV shoplifting dataset (images + keypoint labels).

The dataset's only class is "person", so this teaches the pose model the dataset's look
(overhead store cameras); it does NOT learn "shoplifting". Training/validation are split by
SCENE (whole clips held out) so near-duplicate neighbouring frames can't inflate the score.

Example:
    python3 finetune_pose.py                    # 30 epochs, hold out scene 4
    python3 finetune_pose.py --epochs 50 --device cpu

Output: runs/kaggle_pose/weights/best.pt, copied to yolo11n-pose-kaggle.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil

from ultralytics import YOLO

from pose_utils import MODEL_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def find_dataset() -> str:
    hits = glob.glob(os.path.join(HERE, "kagglehub", "**", "CCTV_Shoplifting_Dataset"), recursive=True)
    if not hits:
        raise SystemExit("Could not find CCTV_Shoplifting_Dataset under ./kagglehub. Pass --dataset PATH.")
    return hits[0]


def build_split(root: str, val_scene: int, out_dir: str) -> str:
    """Write train.txt / val.txt (image paths) and data.yaml; return the yaml path."""
    os.makedirs(out_dir, exist_ok=True)
    train, val = [], []
    for img in sorted(glob.glob(os.path.join(root, "images", "*.png"))):
        scene = int(re.search(r"shoplifting(\d+)_f", os.path.basename(img)).group(1))  # type: ignore[union-attr]
        (val if scene == val_scene else train).append(img)
    with open(os.path.join(out_dir, "train.txt"), "w") as fh:
        fh.write("\n".join(train) + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as fh:
        fh.write("\n".join(val) + "\n")
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(
            f"path: {out_dir}\ntrain: train.txt\nval: val.txt\n"
            f"kpt_shape: [17, 3]\nflip_idx: {COCO_FLIP_IDX}\nnames:\n  0: person\n"
        )
    print(f"Split: {len(train)} train images, {len(val)} val images (scene {val_scene} held out)")
    return yaml_path


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def report(tag: str, m) -> None:
    print(f"{tag}: pose mAP50={m.pose.map50:.3f}  pose mAP50-95={m.pose.map:.3f}  "
          f"box mAP50={m.box.map50:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune yolo11n-pose on the Kaggle dataset.")
    p.add_argument("--dataset", default=None)
    p.add_argument("--val-scene", type=int, default=4, help="Scene (1-4) held out for validation.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=544, help="Multiple of 32; the clips are 544x544.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.0005,
                   help="Initial learning rate (AdamW). Low on purpose: a first run with the auto "
                        "setting (0.002) peaked at epoch 1 and then got worse.")
    p.add_argument("--freeze", type=int, default=10,
                   help="Freeze the first N layers (the backbone) so only the neck/head adapt. 0 = train all.")
    p.add_argument("--device", default=None, help="cpu, mps or a CUDA id; default: auto.")
    p.add_argument("--project", default=os.path.join(HERE, "runs"))
    args = p.parse_args()

    root = args.dataset or find_dataset()
    data = build_split(root, args.val_scene, os.path.join(HERE, "kaggle_pose_data"))
    device = args.device or auto_device()
    print(f"Device: {device}")

    report("BASELINE (original yolo11n-pose, no training)",
           YOLO(MODEL_NAME).val(data=data, imgsz=args.imgsz, device=device, verbose=False, plots=False))

    YOLO(MODEL_NAME).train(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=args.project, name="kaggle_pose", exist_ok=True, workers=2, plots=False, patience=15,
        optimizer="AdamW", lr0=args.lr0, freeze=args.freeze or None,
    )
    best = os.path.join(args.project, "kaggle_pose", "weights", "best.pt")
    report("FINE-TUNED (best.pt)", YOLO(best).val(data=data, imgsz=args.imgsz, device=device, verbose=False, plots=False))
    dest = os.path.join(HERE, "yolo11n-pose-kaggle.pt")
    shutil.copy(best, dest)
    print(f"Saved fine-tuned model to {dest}")


if __name__ == "__main__":
    main()
