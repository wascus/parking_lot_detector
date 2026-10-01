# Parking Lot Detector

Real-time parking spot detection. Each spot is marked as free (green) or occupied (red), with a live count of available spots.

Spots are defined by a mask image (one white rectangle per spot). Each spot is classified as empty or occupied by an SVM trained with scikit-learn.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Runs the detector |
| `util.py` | Shared helpers and the classifier wrapper |
| `train.py` | Trains the classifier and saves it to `model.p` |
| `extract_crops.py` | Extracts spot images to build new training data |
| `clf-data/`, `new-data/` | Training images (`empty/` and `not_empty/`) |

## Getting started

```bash
pip install -r requirements.txt
python train.py
python main.py
```

To adapt the model to a new parking lot, generate crops with `extract_crops.py`, fix any misclassified ones, then retrain with `train.py`.
