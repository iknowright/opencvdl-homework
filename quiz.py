import random
import time
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Subtractor:
    def __init__(self, train_frames):
        self.train(train_frames)

    def train(self, train_frames):
        frames = np.array([
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            for frame in train_frames
        ])

        self.pixels = np.moveaxis(frames, 0, -1)
        height, width, n = self.pixels.shape
        self.mean = np.zeros((height, width))
        self.std = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                pixels = self.pixels[i][j]
                mean = np.mean(pixels)
                std = np.std(pixels)
                std = std if std > 5 else 5
                self.mean[i][j] = mean
                self.std[i][j] = std

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        new_frame = gray - self.mean
        new_frame = (new_frame > 5 * self.std).astype(np.uint8) * 255

        return new_frame


class Quiz1():
    def __init__(self):
        pass

    def background_subtraction(self):
        cap = cv2.VideoCapture('src/Q1_Image/bgSub.mp4')

        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)


        train = frames[:50]
        test = frames[50:]

        model = Subtractor(train)
        # render
        for frame in test:
            subframe = model.apply(frame)
            stacked_subframe = np.stack((subframe,)*3, axis=-1)
            demoframe = cv2.hconcat([frame,stacked_subframe])
            cv2.imshow('Demo', demoframe)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


class Quiz2():
    def __init__(self):
        self.frame = None
        self.points = None

    def get_points(self):
        cap = cv2.VideoCapture('src/Q2_Image/opticalFlow.mp4')

        first = None
        while(cap.isOpened() and first is None):
            ret, frame = cap.read()
            first = frame

        cap.release()

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

        self.points = np.asarray(pts).reshape((len(pts), 1, 2))
        self.frame = frame

    def preprocessing(self):
        self.get_points()
        cv2.imshow("Keypoints", self.frame)
        cv2.waitKey(0)

    def optical_flow(self):
        if self.frame is None:
            self.get_points()

        cap = cv2.VideoCapture('src/Q2_Image/opticalFlow.mp4')
        # skip first frame
        ret, frame = cap.read()
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (21,21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,3)

        # Take first frame and find corners in it
        old_frame = self.frame
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        p0 = self.points.astype(np.float32)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while(1):
            try:
                ret,frame = cap.read()
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

        cap.release()


class Quiz3():
    def __init__(self):
        pass


class Quiz4():
    def __init__(self):
        pass
