import numpy as np
import cv2
import argparse

cap = cv2.VideoCapture('src/featureTracking.mp4')

ret, old_frame = cap.read()
frame = old_frame.copy()

# Tune the parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.85

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(frame)

pts = []
for i in keypoints:
    x,y = i.pt
    x = round(x)
    y = round(y)
    if x < 500:
        pts.append(i.pt)
        frame = cv2.rectangle(frame, (x-5, y-5), (x+5,y+5), (0, 0, 255), 1) 

points = np.asarray(pts).reshape((len(pts), 1, 2))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (21,21),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,3)
# Take first frame and find corners in it
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = points.astype(np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    try:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        break

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color.tolist(), 1)
        a = int(round(a))
        b = int(round(b))
        frame = cv2.rectangle(frame, (a-5, b-5), (a+5,b+5), (0, 0, 255), 1) 

    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)