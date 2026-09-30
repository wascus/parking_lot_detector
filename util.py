import cv2
import numpy as np
import pickle


MODEL = pickle.load(open("model.p", "rb"))

def get_parking_spots_bboxes(connected_components):
    """
    Extracts bounding boxes for parking spots from connected components.

    Args:
        connected_components: A tuple containing the number of labels, label matrix, stats, and centroids.
    Returns:
        List of bounding boxes for each parking spot.
    """
    num_labels, labels, stats, centroids = connected_components
    bboxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, width, height))
    return bboxes

def preprocess_spot(spot_bgr):
    """
    Turns a spot crop into the feature vector the model expects.
    Shared by is_empty and train.py so training and inference always match.
    """
    # the model was trained on horizontal spots (cars lying sideways): rotate vertical ones to match
    height, width = spot_bgr.shape[:2]
    if height > width:
        spot_bgr = cv2.rotate(spot_bgr, cv2.ROTATE_90_CLOCKWISE)

    # the model was trained on skimage-style images: float values in [0, 1]
    img_resized = cv2.resize(spot_bgr, (15, 15)) / 255.0
    return img_resized.flatten()

def is_empty(spot_bgr):
    flat_data = np.array([preprocess_spot(spot_bgr)])

    y_output = MODEL.predict(flat_data)

    return y_output[0] == 0