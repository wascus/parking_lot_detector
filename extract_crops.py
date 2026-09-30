"""
Cuts every parking spot out of frames spread across a video, to build training data for that camera.

Crops are pre-sorted with the current model into <out>/empty and <out>/not_empty.
Review both folders and move the misclassified crops into the right one, then run train.py.

Usage:
    python extract_crops.py
    python extract_crops.py --video samples/parking_lot_extended.mp4 --mask samples/parking_lot_mask.png --frames 8
"""
import argparse
import os

import cv2
import numpy as np

from util import get_parking_spots_bboxes, is_empty, preprocess_spot


parser = argparse.ArgumentParser()
parser.add_argument("--video", default=os.path.join("samples", "parking_lot_extended.mp4"))
parser.add_argument("--mask", default=os.path.join("samples", "parking_lot_mask.png"))
parser.add_argument("--out", default="new-data")
parser.add_argument("--frames", type=int, default=8, help="number of frames sampled evenly across the video")
parser.add_argument("--min-change", type=float, default=0.05,
                    help="skip a spot if it looks the same as the last saved crop (mean abs difference, 0-1)")
parser.add_argument("--force", action="store_true", help="write into --out even if it already contains crops")
args = parser.parse_args()

# don't mix new crops into folders that may already hold hand-labelled ones
for label in ["empty", "not_empty"]:
    folder = os.path.join(args.out, label)
    if os.path.isdir(folder) and os.listdir(folder) and not args.force:
        raise SystemExit(f"{folder} already contains files (maybe your labels). Use --force or pick another --out.")
    os.makedirs(folder, exist_ok=True)

mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise SystemExit(f"Can't read mask {args.mask}")
bboxes = get_parking_spots_bboxes(cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S))

cap = cv2.VideoCapture(args.video)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if n_frames <= 0:
    raise SystemExit(f"Can't read video {args.video}")

last_saved = {}
counts = {"empty": 0, "not_empty": 0, "skipped": 0}
for frame_idx in np.linspace(0, n_frames - 1, args.frames).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    for spot_idx, (x, y, width, height) in enumerate(bboxes):
        spot_bgr = frame[y:y + height, x:x + width]
        features = preprocess_spot(spot_bgr)

        if spot_idx in last_saved and np.abs(features - last_saved[spot_idx]).mean() < args.min_change:
            counts["skipped"] += 1
            continue
        last_saved[spot_idx] = features

        label = "empty" if is_empty(spot_bgr) else "not_empty"
        cv2.imwrite(os.path.join(args.out, label, f"f{frame_idx:05d}_s{spot_idx:03d}.png"), spot_bgr)
        counts[label] += 1

cap.release()
print(f"{len(bboxes)} spots, {args.frames} frames")
print(f"saved {counts['empty']} as empty, {counts['not_empty']} as not_empty "
      f"(skipped {counts['skipped']} unchanged crops) in {args.out}")
print("Now review both folders, move misclassified crops, then run train.py")
