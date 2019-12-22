import numpy as np
import cv2

cap = cv2.VideoCapture('src/bgSub.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)

while(cap.isOpened()):
    ret, frame = cap.read()

    try:
        subframe = fgbg.apply(frame)
        cv2.imshow('Original', frame)
        cv2.moveWindow('Original',250,200)
        cv2.imshow('Subtracted', subframe)
        cv2.moveWindow('Subtracted',600,200)
    except:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()