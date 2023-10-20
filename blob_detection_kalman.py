import cv2
import numpy as np
from kalmanfilter import KalmanFilter


cap = cv2.VideoCapture(0)
detector = cv2.SimpleBlobDetector_create()

kf = KalmanFilter()

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # detecting the frame
        keypoints = detector.detect(frame)

        for keypoint in keypoints:
            # i is the index of the blob you want to get the position
            x = keypoint.pt[0]
            y = keypoint.pt[1]
            print(x, y)
            predicted = kf.predict(x, y)
            cv2.circle(
                frame, (predicted[0], predicted[1]), 20, (255, 0, 0), -1)

            # predicted2 = kf.predict(predicted[0], predicted[1])
            # cv2.circle(
            #     frame, (predicted2[0], predicted2[1]), 20, (255, 255, 0), -1)

        # drawing detected keypoints as circles
        img_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("result", img_keypoints)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
