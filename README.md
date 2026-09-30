# Parking Lot Detector

Counts free parking spots in a video of a parking lot. Each spot is marked green (empty) or red (occupied), and a live counter shows how many spots are available.

It works in two steps:

1. **Find the spots.** A black-and-white mask image marks each parking spot as a white rectangle. OpenCV's connected-components analysis turns every white region into a bounding box.
2. **Classify each spot.** On every frame, each spot is cropped, resized to 15×15 pixels and passed to an SVM classifier (scikit-learn) that predicts `empty` or `not_empty`.

## Project structure

| File / folder | Purpose |
|---|---|
| `main.py` | Plays the video with spots and the available-spots counter drawn on top. Press `q` to quit. |
| `util.py` | Shared helpers: mask → bounding boxes, crop preprocessing, and `is_empty()`, which uses the trained model. |
| `train.py` | Trains the classifier on labelled crops and saves it to `model.p`. |
| `extract_crops.py` | Cuts spot crops out of a video to build new training data for a camera. |
| `clf-data/`, `new-data/` | Training images, each split into `empty/` and `not_empty/` subfolders. |

## Setup

Requires Python 3.

```bash
python -m venv venv
venv\Scripts\activate          # on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install icecream
```

### Files not included in the repo

The video, the mask and the trained model are **not** in this repository, because they are too large or can be regenerated. You need to provide them:

- `samples/parking_lot_extended.mp4`: the parking lot video
- `samples/parking_lot_mask.png`: the mask, with the same resolution as the video, a black background and one white rectangle per parking spot
- `model.p`: create it by running `python train.py` (see below)

## Usage

**1. Train the model.** This uses the images in `clf-data/` and `new-data/`:

```bash
python train.py
```

It tries several SVM settings, prints the test accuracy and a confusion matrix, and saves the best model to `model.p`. If a `model.p` already exists, it is first copied to `model_backup.p`.

**2. Run the detector:**

```bash
python main.py
```

## Adapting to a new camera

A model trained on one parking lot may not work well on another. To add training data from a new video:

1. Draw a mask for the new video (white rectangles on black, same resolution as the video).
2. Extract crops. They are pre-sorted into `empty/` and `not_empty/` using the current model:

   ```bash
   python extract_crops.py --video samples/my_video.mp4 --mask samples/my_mask.png --out new-data --frames 8
   ```

   Crops that barely change between frames are skipped, so each spot is saved only when it looks different. The script refuses to write into folders that already contain files unless you pass `--force`.
3. Check both folders and move any misclassified crops into the correct one.
4. Retrain with `python train.py`. Use `--data` to choose which folders to train on.

Crops of vertical spots are rotated to horizontal before classification, so the same model handles both orientations.
