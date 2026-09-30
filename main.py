import cv2
from util import *
mask_path = ".\samples\parking_lot_mask.png"
video_path = ".\samples\parking_lot_extended.mp4"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Mask', mask)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

bboxes = get_parking_spots_bboxes(connected_components)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    available = 0
    for bbox in bboxes:
        x, y, width, height = bbox
        spot_bgr = frame[y:y + height, x:x + width]
        empty = is_empty(spot_bgr)
        if empty:
            available += 1

        color = (0, 255, 0) if empty else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

    # show frame resized to fit the screen
    frame = cv2.resize(frame, (960, 540))

    # draw the counter after resizing so the text size doesn't depend on the video resolution
    counter_text = f"Available spots: {available} / {len(bboxes)}"
    cv2.rectangle(frame, (10, 10), (330, 50), (0, 0, 0), -1)
    cv2.putText(frame, counter_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()