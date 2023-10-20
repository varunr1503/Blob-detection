# https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np
from kalmanfilter import KalmanFilter
from collections import defaultdict


def rescaleFrame(frame, scale=0.5):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture("blob1.mp4")

track_history = defaultdict(lambda: [])

# Load detector
detector = cv2.SimpleBlobDetector_create()
# Load Kalman filter to predict the trajectory
kf = KalmanFilter()

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    frame = rescaleFrame(frame)

    keypoints = detector.detect(frame)
    i = 0
    for keypoint in keypoints:
        # i is the index of the blob you want to get the position
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        # print(x, y)
        # print(keypoint)
        predicted = kf.predict(x, y)

        cv2.circle(
            frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)
        cv2.circle(
            frame, (int(x), int(y)), 20, (0, 0, 255), 4)

        predicted2 = kf.predict(predicted[0], predicted[1])
        cv2.circle(
            frame, (predicted2[0], predicted2[1]), 20, (255, 255, 0), 4)

        track = track_history[i]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 5:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(
            230, 230, 230), thickness=10)
        i = i+1

    # drawing detected keypoints as circles
    # img_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
    #     []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("result", frame)

    key = cv2.waitKey(150)
    if key == 27:
        break
