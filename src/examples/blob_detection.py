# Standard imports
import cv2
import numpy as np

cap = cv2.VideoCapture('src/featureTracking.mp4')

first = None
while(cap.isOpened() and first is None):
    ret, frame = cap.read()
    first = frame
    
cap.release()

# cv2.imshow('first', frame)

# Tune the parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.85

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(frame)

array = np.array([])
points = []
for i in keypoints:
    x,y = i.pt
    x = round(x)
    y = round(y)
    if x < 500:
        points.append(i.pt)
        frame = cv2.rectangle(frame, (x-5, y-5), (x+5,y+5), (0, 0, 255), 1) 

    point = np.array([x, y])
    array = np.append(array, point)
    

array = array.reshape((8,1,2))
print(array)

print(np.asarray(points))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(frame, points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
cv2.imshow("Keypoints", frame)
cv2.waitKey(0)