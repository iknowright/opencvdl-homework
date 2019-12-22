from PyQt5.QtWidgets import QMessageBox
# Image processing use case
import cv2
import matplotlib.pyplot as plt
import numpy as np

import random
import time

class Quiz1():
    def __init__(self):
        self.imageL = cv2.imread('src/imL.png', 0)
        self.imageR = cv2.imread('src/imR.png', 0)

    def depth_map(self):
        stereo = cv2.StereoSGBM_create(
            minDisparity = 0,
            numDisparities = 64,
            blockSize = 9,
            P1 = 16*3*9,
            P2 = 64*3*9,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )
        stereo.setMinDisparity(0)
        disparity = stereo.compute(self.imageL, self.imageR)
        plt.imshow(disparity, 'gray')
        plt.show()
        cv2.imshow('left', self.imageL)


class Quiz2():
    def backgroundSubtraction(self):
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

class Quiz3():
    def __init__(self):
        self.frame = None
        self.points = None

    def get_points(self):
        cap = cv2.VideoCapture('src/featureTracking.mp4')

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

    def opticalFlow(self):
        if self.frame is None:
            self.get_points()

        cap = cv2.VideoCapture('src/featureTracking.mp4')
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

class Quiz4():
    def __init__(self):
        pass
    
    def projection(self):
        intrinsic = np.array(
            [[2225.49585482, 0, 1025.5459589],
            [0, 2225.18414074, 1038.58518846],
            [0, 0, 1]])

        distortion = np.array(
            [[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]])

        params = []

        bmp1 = np.array(
            [[-0.97157425, -0.01827487, 0.23602862, 6.81253889],
            [0.07148055, -0.97312723, 0.2188925, 3.37330384],
            [0.22568565, 0.22954177, 0.94677165, 16.71572319]])
        params.append(bmp1)

        bmp2 = np.array(
            [[-0.8884799, -0.14530922, -0.435303, 3.3925504],
            [0.07148066, -0.98078915, 0.18150248, 4.36149229],
            [-0.45331444, 0.13014556, 0.88179825, 22.15957429]])
        params.append(bmp2)

        bmp3 = np.array(
            [[-0.52390938, 0.22312793, 0.82202974, 2.86774801],
            [0.00530458, -0.96420621, 0.26510046, 4.70990021],
            [0.85175747, 0.14324914, 0.50397308, 12.98147662]])
        params.append(bmp3)

        bmp4 = np.array(
            [[-0.63108673, 0.53013053, 0.566296, 1.22781875],
            [0.13263301, -0.64553994, 0.75212145, 3.48023006],
            [0.76428923, 0.54976341, 0.33707888, 10.9840538]])
        params.append(bmp4)

        bmp5 = np.array(
            [[-0.87676843, -0.23020567, 0.42223508, 4.43641198],
            [0.19708207, -0.97286949, -0.12117596, 0.67177428],
            [0.43867502, -0.02302829, 0.89835067, 16.24069227]])
        params.append(bmp5)

        pyramid = np.array(
            [[3,3,-4],[1,1,0],[1,5,0],[5,5,0],[5,1,0]]).astype(np.float32)

        # red color
        red  = (0,0,255)

        for i in range(5):
            # rvec and tvec
            rotation_v = params[i][:,:3]
            translation_v = params[i][:,3]

            # 3d to 2d mapping points
            pts, _ = cv2.projectPoints(pyramid, rotation_v, translation_v, intrinsic, distortion)

            # input image
            img = cv2.imread('src/{}.bmp'.format(i+1))

            # base square
            img = cv2.line(img, (pts[1][0][0],pts[1][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
            img = cv2.line(img, (pts[2][0][0],pts[2][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)
            img = cv2.line(img, (pts[3][0][0],pts[3][0][1]),(pts[4][0][0],pts[4][0][1]), red, 10)
            img = cv2.line(img, (pts[4][0][0],pts[4][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)

            # side lines to vertex
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[4][0][0],pts[4][0][1]), red, 10)

            # resize to view
            resize = cv2.resize(img, (600,600))
            cv2.imshow('projection', resize)
            time.sleep(0.5)
            cv2.waitKey(30)

        cv2.waitKey(0)