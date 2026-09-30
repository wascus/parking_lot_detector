"""
Trains the empty / not-empty spot classifier and saves it to model.p.

Each data folder must contain an 'empty' and a 'not_empty' subfolder of spot crops.
Images go through the same preprocess_spot as is_empty, so training and inference match.

Usage:
    python train.py
    python train.py --data clf-data new-data --out model.p
"""
import argparse
import os
import pickle
import shutil

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from util import preprocess_spot


LABELS = {"empty": 0, "not_empty": 1}

parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="+", default=["clf-data", "new-data"])
parser.add_argument("--out", default="model.p")
args = parser.parse_args()

data, labels, sources = [], [], []
for data_dir in args.data:
    for label_name, label in LABELS.items():
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            print(f"warning: {folder} not found, skipping")
            continue
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            if img is None:
                continue
            data.append(preprocess_spot(img))
            labels.append(label)
            sources.append(data_dir)
    print(f"{data_dir}: {sum(s == data_dir for s in sources)} images")

data, labels, sources = np.array(data), np.array(labels), np.array(sources)
if len(np.unique(labels)) < 2:
    raise SystemExit("Need both empty and not_empty images to train.")

x_train, x_test, y_train, y_test, src_train, src_test = train_test_split(
    data, labels, sources, test_size=0.2, shuffle=True, stratify=labels, random_state=0)

classifier = GridSearchCV(SVC(), [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}], n_jobs=-1)
classifier.fit(x_train, y_train)
model = classifier.best_estimator_
print("best params:", classifier.best_params_)

y_pred = model.predict(x_test)
print(f"test accuracy (all): {accuracy_score(y_test, y_pred):.2%}")
for data_dir in args.data:
    sel = src_test == data_dir
    if sel.any():
        print(f"test accuracy ({data_dir}): {accuracy_score(y_test[sel], y_pred[sel]):.2%} on {sel.sum()} images")
print("confusion matrix (rows = true empty/not_empty, cols = predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

if os.path.exists(args.out):
    backup = os.path.splitext(args.out)[0] + "_backup.p"
    shutil.copy(args.out, backup)
    print(f"previous model backed up to {backup}")
with open(args.out, "wb") as f:
    pickle.dump(model, f)
print(f"model saved to {args.out}")
