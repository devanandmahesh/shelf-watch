# ShelfWatch

Shoplifting detection from a camera. It tracks every person, scores each one from their body motion
and from the pixels around them, and flags whoever looks like they are concealing something. **An
alert is a prompt for a human to look, never proof of anything** — and on a camera the models were
not trained on, they are barely better than guessing, which this README documents rather than hides.

```bash
pip install -r requirements.txt
python3 web_app.py        # browser UI at http://127.0.0.1:8000
python3 demo.py           # or an OpenCV window
```

Three models ship with it, each stage an attempt to fix the one before — see
[which model](#which-model-and-why-are-there-several) — and two front ends that share one alert
rule: a browser UI and an OpenCV window. Everything uses `yolo11n-pose.pt` for 17 body keypoints
per person, and the video stages add a frozen MViT-V2-S for the pixels around each person.

## What is in this repository, and what is not

Every script, the page, and **every model the commands below load**, so a fresh clone runs each
section without a training step first:

| ships | size | loaded by |
|---|---|---|
| `video_model.pkl` | 24 KB | `demo.py` / `web_app.py` by default |
| `fused_model.pkl` | 37 MB (a 200-tree forest) | `--model fused_model.pkl` |
| `kaggle_window_model.pkl` | 1.2 MB | `--model kaggle_window_model.pkl` |
| `activity_model.pkl`, `kaggle_model.pkl`, `kaggle_full_model.pkl` | ~2.3 MB | the single-frame sections |
| `dataset.csv`, `kaggle_dataset.csv`, `kaggle_full_dataset.csv` | 0.6 MB | `train_model.py` |

Left out, because it is bulky and the scripts here rebuild it:

| not included | how to get it |
|---|---|
| `yolo11n-pose.pt` | downloaded automatically on the first run |
| the Kaggle CCTV dataset | `python3 data.py`, or `kagglehub` on first use |
| `archive/` videos (one sub-folder per class) | your own footage, or any shoplifting/normal video set |
| `archive_windows.csv` (67 MB), `kaggle_windows.csv` (3 MB) | `folder_to_windows.py`, `kaggle_to_windows.py` |
| `video_embeddings.npz` (9 MB) | `videos_to_embeddings.py` |
| `archive_only_window_model.pkl`, `shoplift_window_model.pkl` (37 MB each) | `train_window_model.py` with those CSVs |

Retraining on your own footage is the point, not an afterthought — the numbers below come from 178
archive videos plus 8 Kaggle clips, and a model that has never seen your camera is barely better
than guessing on it.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

On macOS the first camera use asks for permission. Allow it for the app you run Python from
(Terminal, iTerm, VS Code). If you see "not authorized to capture video", enable it in
System Settings > Privacy & Security > Camera, then restart that app.

## Files

| File | Purpose |
|---|---|
| `README.md` | This file. |
| `requirements.txt` | The five packages this needs (plus torch for the video model). |
| `.gitignore` | Keeps datasets and derived CSVs out of git; the trained models are let through. |
| `collect_data.py` | Record labelled keypoints from a webcam or video into a CSV. |
| `train_model.py` | Train a RandomForest on that CSV and save a `.pkl`. |
| `live_inference.py` | Classify live from a webcam or video file; raises a verdict alert. |
| `kaggle_to_csv.py` | Convert the Kaggle CCTV dataset into a single-frame training CSV, and read its concealment labels. |
| `define_zones.py` | Draw shelf zones on a frame; saves `zones.json`. |
| `risk_engine.py` | Rule engine: reach into a shelf zone, then hand to body, builds a risk score. |
| `shoplift_demo.py` | Tracking + zones + risk score + dashboard, alerts, snapshots, event log. |
| `finetune_pose.py` | Fine-tune YOLO pose on the Kaggle dataset's own keypoint labels. |
| `data.py` | Downloads the Kaggle dataset with `kagglehub`. |
| `pose_utils.py` | Shared keypoint extraction. Always returns exactly 34 floats (17 x,y pairs), normalised to the person's bounding box; undetected keypoints are 0.0. |
| `demo.py` | Camera demo: tracks every person, scores each separately, flags the suspect with a red box. Loads `video_model.pkl` by default. `python3 demo.py` |
| `video_utils.py` | Video-model helpers: person crops over 2 s, the frozen MViT-V2-S backbone, live scoring. |
| `videos_to_embeddings.py` | Embed the archive + Kaggle videos into clip embeddings (`video_embeddings.npz`). |
| `train_video_model.py` | Train and honestly test the video model; saves `video_model.pkl` for `demo.py`. |
| `fused_utils.py` | Runs both streams for one person and averages their scores behind one threshold. |
| `train_fused_model.py` | Train the fused pose + video model; saves `fused_model.pkl` for `demo.py`. |
| `engine.py` | The alert rule and the drawing shared by `demo.py` and `web_app.py`. |
| `web_app.py` | Browser front end: live view, per-person risk, alert log, live threshold. `python3 web_app.py` |
| `web_ui.html` | The page `web_app.py` serves. |
| `window_utils.py` | Time-window features: a person's pose sampled over the last ~1.5 s (340 numbers, incl. motion). |
| `kaggle_to_windows.py` | Build the time-window training CSV (`kaggle_windows.csv`) from the Kaggle videos. |
| `folder_to_windows.py` | Build the same CSV from a folder of your own videos, one sub-folder per class. |
| `train_window_model.py` | Train the time-window model with a leave-one-scene-out test; saves `kaggle_window_model.pkl` for `demo.py`. |

## Which model, and why are there several?

Each section below was an attempt to fix the one before it. They are not alternatives to pick
between so much as a trail, and knowing the trail is how you read the numbers:

| stage | input it reads | verdict |
|---|---|---|
| single-frame pose | 34 numbers, one pose | a still pose cannot show an action; dropped |
| pose window | 340 numbers, ~1.5 s of motion | better, but keypoints carry no item, bag or hand detail |
| video | 768 numbers, 2 s of pixels | **`video_model.pkl`, the one that ships** |
| fused | both of the last two | one file, one threshold; the best of them on its own camera |

The first stage is history: its scripts and checkpoints are not in this repo, only the lesson. The
pose-window stage still earns its keep as half of the fused model, so `kaggle_to_windows.py`,
`folder_to_windows.py` and `train_window_model.py` are here.

`demo.py` and `web_app.py` both load `video_model.pkl` unless you pass `--model`, because it is the
livelier one on a camera nothing was trained on, which is the usual case when trying this out. On the
camera it *was* trained on, the fused model catches more at the same false-alarm rate (78/92 against
71/92, section 3), so build it with `train_fused_model.py` and pass `--model fused_model.pkl` once
you have calibrated a fixed camera. On an unseen camera the two differ only in where the threshold
cuts, not in accuracy — see
["the video model detects more"](#the-video-model-detects-more-than-the-fused-one).

## 1. Your own activity classifier

Record at least two classes (for example `idle` and `waving`) on the same camera you will use live.

```bash
python3 collect_data.py --label idle
python3 collect_data.py --label waving
```

- Click the video window, then **hold `r`** to record. Press `q` to quit.
- Aim for about 500+ frames per class; vary distance, angle and speed.
- Use `--source clip.mp4` to process a whole video automatically. `--output` sets the CSV (default `dataset.csv`).
- Each run appends to the CSV, so you can record several sessions.

Train, then run live:

```bash
python3 train_model.py
python3 live_inference.py
```

`train_model.py` options: `--input`, `--output` (default `activity_model.pkl`), `--max-missing`
(drop rows with more than N of 17 keypoints undetected, default 8), `--test-size`, `--trees`, `--seed`.

`live_inference.py` options: `--model`, `--source` (webcam index or video path), `--alert-class`
(class that counts as suspicious, default `conceal`), `--alert-frames` (default 8), `--no-display`.
When the alert class holds for `--alert-frames` frames in a row, a red banner appears and a final
verdict is printed at the end.

Notes:
- Don't mirror/flip the camera in any script; it would swap left and right keypoints.
- The "held `r`" key is inferred from OS key-repeat, so recording stops about 0.15 s after release.
  Tune `FIRST_PRESS_GRACE_S` / `REPEAT_GRACE_S` at the top of `collect_data.py` if it feels off.
- The accuracy printed by `train_model.py` uses a random split of frames, which is optimistic
  because neighbouring frames are near-identical. Test on a separate recording for a real number.

## 2. Rule-based shoplifting demo (no training)

Draw the shelf zones once on your footage, then run the demo:

```bash
python3 define_zones.py --source store.mp4
python3 shoplift_demo.py --source store.mp4
```

Per tracked person (ByteTrack IDs) it scores "wrist enters a shelf zone, then hand to the body within a
few seconds". Scores decay over time; crossing the threshold raises an alert: the person is
spotlighted, a head zoom is shown, a snapshot is saved to `alerts/`, and events go to `events.csv`
and a side dashboard.

Options: `--threshold` (default 40), `--window` (4 s), `--pocket-ratio`, `--decay`,
`--conceal-region hip|torso`, `--save-video out.mp4`, `--no-display`, `--no-face-zoom`.

**Accuracy is poor.** On the 8 Kaggle clips it missed the real concealments and gave false alarms on
normal clips (best case 1 of 4 shoplifting clips caught, 3 of 4 normal clips falsely flagged). Pose alone
cannot tell hiding an item from holding it. Treat every alert as "a person should look", never as proof.
A bag cue from a COCO detector (handbag/backpack) was tried and made it worse, so it is not enabled.

## 3. Training on the Kaggle shoplifting dataset

Dataset: `simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm` (downloaded into `./kagglehub`).
It has 456 frames from 8 short synthetic clips (4 shoplifting, 4 normal, 4 scenes).

```bash
python3 kaggle_to_csv.py
python3 train_model.py --input kaggle_dataset.csv --output kaggle_model.pkl
python3 live_inference.py --source kagglehub/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm/versions/1/CCTV_Shoplifting_Dataset/videos/shoplifting1.mp4 --model kaggle_model.pkl
```

Frames are labelled `conceal` if they fall in the concealment segment described in the dataset's VLM
JSON (shoplifting clips only), else `normal`. Keypoints are re-extracted with our own `yolo11n-pose`
(the default). `--use-labels` would use the dataset's own keypoint labels instead, but those come from a
different pose model, so the classifier then misreads live keypoints; don't use it.

**Limits (measured):**

| Test | Result |
|---|---|
| Trained on all 8 clips, verdict on those same clips | 8 of 8 correct |
| Train on 3 scenes, verdict on the 4th (each scene) | 4 of 8 correct: all normal clips cleared, every shoplifting clip missed |

The dataset is too small and too uniform for the model to learn the behaviour instead of the four
scenes, so it will not work on new cameras or footage. Use it to demo the pipeline, not as a detector.

## 4. Time-window model (uses motion, not a single pose)

```bash
python3 demo.py --model kaggle_window_model.pkl      # the model ships; these rebuild it:
python3 kaggle_to_windows.py && python3 train_window_model.py
```

It classifies each tracked person's last ~1.5 s of keypoints. Measured with whole scenes held out
(AUC, 0.5 = guessing): single-frame model 0.45 on average, window model 0.65. Hard verdicts are
still 5 of 8 clips correct for both, because four scenes are too few. `demo.py` detects the model
type automatically (window models are saved as a dict).

## 5. Video model (looks at the pixels, not just the skeleton)

```bash
python3 videos_to_embeddings.py     # ~180 videos -> clip embeddings (slow, one pass)
python3 train_video_model.py
python3 demo.py --model video_model.pkl --source archive/shoplifting/shoplifting-20.mp4
```

Each tracked person's last 2 s is cropped and embedded by a frozen MViT-V2-S (Kinetics-400); a
logistic regression on those 768 numbers scores the clip, and a person's score is the 90th
percentile of their recent clips. Measured with whole videos held out:

| test | clip AUC | video AUC | caught | falsely flagged |
|---|---|---|---|---|
| archive, videos held out | 0.88 | 0.92 | 83/92 | 9/86 normal |
| Kaggle, camera never seen | 0.60 | 0.56 | 3/4 | 3/4 normal |

**Read the second row before trusting an alert.** On the camera it was trained on the model is
good; on a camera it has never seen it is barely better than guessing *and* its scores sit higher,
so the saved threshold (0.85, chosen to flag at most 10% of held-out normal archive videos) fires
on most normal footage there. A webcam pointed at your room is such an unseen camera.

To use it on another camera, record a few minutes of ordinary activity there and let the demo fit
the threshold to it:

```bash
python3 demo.py --model video_model.pkl --calibrate --source my_quiet_shop.mp4
python3 demo.py --model video_model.pkl --threshold 0.97        # whatever it suggested
```

`--calibrate` raises no alerts; it reports the highest windowed risk each tracked person reached on
footage you know is theft-free, and the threshold that would have kept them all quiet. Raising the
threshold costs real detections, so the honest reading is how high it has to go.

Live, a person needs ~8 s of tracking before any verdict, and the verdict is read off their last
~10 s only, so it clears again when the risk falls back. Note that a live camera re-tests every
0.5 s where the table above tests each video once, so a given false-alarm rate per video becomes a
much higher chance over minutes of watching: replaying the held-out archive scores through the live
rule, a normal person trips it about once per minute of being watched, on the camera it was trained
on.

## 6. Fused model: one file, one threshold

```bash
python3 train_fused_model.py
python3 demo.py --model fused_model.pkl
```

Both streams score every tracked person — the pose window from their skeleton motion, the video
model from the pixels — and the two 90th percentiles are averaged into one number against one
threshold (`fused_utils.py`). The pose features cost nothing extra live, because `demo.py` already
runs YOLO pose tracking to get the boxes.

Whole videos held out (5 folds, same folds and same videos for every row). Each stream is judged at
its own threshold for the same false-alarm budget, 9 of 86 normal videos, because a stream that
simply scores lower is not thereby better:

| stream | archive AUC | archive caught | Kaggle AUC (camera never seen) |
|---|---|---|---|
| pose window only | 0.882 | 59/92 | 0.562 |
| video only | 0.905 | 71/92 | 0.562 |
| **fused** | **0.918** | **78/92** | 0.562 |

In-domain the fusion is worth having: 7 more of 92 shoplifting videos caught than the video stream
alone, at the same false-alarm rate. The AUC gain looks tiny (+0.013) because AUC averages over all
thresholds; the operating point is where it shows. Do not compare these catch counts with section
5's, which used a different fold split — a resplit moves them by several videos either way.

On a camera it has never seen, fusing changes nothing: all three stay at 0.562, i.e. chance. The
two streams' per-video scores correlate 0.75, which is the explanation — they are largely reading
the same thing, most likely the scene rather than the behaviour. So fuse for the in-domain gain,
one model file and one alert rule, not for accuracy you can carry to another camera.

### "The video model detects more than the fused one"

True, off its own camera — and it is not because it is more accurate. Both sit at 0.562 AUC on the
Kaggle camera, i.e. neither tells theft from normal there. What differs is how often they fire, at
the thresholds each one ships with:

| model | on the unseen camera | catches | false alarms |
|---|---|---|---|
| `video_model.pkl` (thr 0.85) | fires on 6 of 8 videos | 3/4 | 3/4 |
| `fused_model.pkl` (thr 0.83) | fires on 2 of 8 videos | 1/4 | 1/4 |

The pose head scores everything low on a camera it has never seen, so averaging drags the fused
number under its threshold. You are not seeing better detection, you are seeing a more trigger-happy
threshold — the two agree on which videos are *most* suspicious, they just cut the list at a
different place. If you want the demo to react more on an uncalibrated camera, lower the threshold
(`--threshold 0.6`) rather than changing model; it is the same trade either way.

The fused model also used to go quiet whenever a person's keypoints dropped out, because it waited
for both streams. It now falls back to the video stream alone, at that stream's own threshold, so a
missing skeleton costs accuracy instead of silencing the model. `--verbose` says
`(video stream only)` when a person is being judged that way.

### Live camera: why a video file works and the webcam seems dead

Two things differ live, and both show up as "it never flags anything":

```bash
python3 demo.py --verbose          # per-person progress, once a second
python3 demo.py --min-clips 6      # verdict after ~5 s instead of ~8 s
```

`--verbose` prints, for each tracked person, how many crops and clip scores they have and each
stream's score, so you can see which stage is stuck: no people means YOLO is not detecting anyone,
`0/12 clip scores` for more than a few seconds means clips are not being built, and a low score
with full counts means the model simply does not find you suspicious. If no clip is scored within
15 s, the demo says so and why.

- **A verdict needs ~8 s of unbroken tracking** (12 clip scores, 0.5 s apart, after 2 s of filling).
  Walk out of frame or let ByteTrack renumber you and the count starts over, so on a webcam you may
  never reach it while you do reach it easily in a 11 s video. `--min-clips 6` halves the wait.
- **A 2 s clip needs the loop above ~2.5 fps** and no tracking gap longer than 0.4 s. A large camera
  frame on a busy machine can fall under that, and then no clip is ever valid.

Also: a video from `archive/` or the Kaggle set is not a test. The shipped model was trained on all
of them, so near-perfect results there are a fit, not a measurement — the honest figures are the
held-out ones in the tables above, and on your own camera they are the Kaggle column: chance.

## 7. Browser front end

```bash
python3 web_app.py                                   # video model, webcam, port 8000
python3 web_app.py --model fused_model.pkl           # pose + pixels, for a calibrated camera
python3 web_app.py --source archive/shoplifting/shoplifting-20.mp4 --loop
```

Then open <http://127.0.0.1:8000>. It serves the same annotated view as `demo.py` plus what the
OpenCV window cannot show:

- a card per tracked person — risk bar with the threshold marked, and the progress line that
  `--verbose` prints, so you can see who is still collecting history
- an alert log, with the risk and threshold each alert was raised or cleared at
- **the threshold and the warm-up as live sliders**, so you can calibrate against your own camera
  without restarting, and a "calibrating" switch that measures without alerting
- a calibration card: the highest risk anyone reached, how many would have been flagged, and the
  threshold that would keep this footage quiet

It needs no extra packages — the server is `http.server` from the standard library, and the page is
plain HTML and CSS. One capture thread owns the models, the slow video network runs on one worker
thread as in `demo.py`, and the browser gets an MJPEG stream plus `GET /api/state` JSON. `demo.py`
and `web_app.py` share `engine.py`, so the alert rule and the drawing are the same code in both.

It binds to `127.0.0.1`, this machine only. `--host 0.0.0.0` exposes the camera, with no
authentication, to anyone who can reach the port — only on a network you trust.

## Output files

`*_windows.csv` and `video_embeddings.npz` (training data), `video_model.pkl` / `fused_model.pkl`
(trained models), and whatever `demo.py --save-video` writes. All of it is ignored by git except the
shipped `video_model.pkl`.
