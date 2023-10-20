import cv2
import numpy as np


def rescaleFrame(frame, scale=0.17):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


img = cv2.imread('dice.png', cv2.IMREAD_GRAYSCALE)

img = rescaleFrame(img)

detector = cv2.SimpleBlobDetector_create()

# detecting the img
keypoints = detector.detect(img)
print(keypoints)

# drawing detected keypoints as circles
img_keypoints = cv2.drawKeypoints(img, keypoints, np.array(
    []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# displaying
cv2.imshow("result", img_keypoints)
cv2.waitKey(0)

cv2.destroyAllWindows()
