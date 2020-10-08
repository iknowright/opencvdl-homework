import random
import time
import glob

from PyQt5.QtWidgets import QMessageBox
# Image processing use case
import cv2
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(suppress=True, formatter={'float_kind':'{:.3f}'.format})


class Quiz1():
    def __init__(self):
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane
        self.image_size = 0
        self.cb_width = 11
        self.cb_height = 8
        self.filenames = [f'src/Q1_Image/{i+1}.bmp' for i in range(15)]
        self.instrinc_matrix = None
        self.dist = None
        self.current_filename = ''

    def chessboard_corners(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_point = np.zeros((self.cb_width * self.cb_height, 3), np.float32)
        obj_point[:,:2] = np.mgrid[0 : self.cb_width, 0 : self.cb_height].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        images = glob.glob('src/Q1_Image/*.bmp')


        axes=[]
        fig=plt.figure()

        for filename in self.filenames:
            img = cv2.imread(filename)

            height, width, channels = img.shape

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cb_width, self.cb_height), None)
            # If found, add object points, image points (after refining them)
            # print(ret, corners)
            if ret == True:
                self.obj_points.append(obj_point)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                self.img_points.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.cb_width, self.cb_height), corners2, ret)

                img = cv2.resize(img, (1024, 1024))                    # Resize image
                cv2.imshow(filename, img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        self.image_size = gray.shape[::-1]


    def get_intrinsic_matrix(self):
        ret, mtx, dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size, None, None)
        if ret:
            self.instrinc_matrix = mtx
            self.dist = dist
            print(mtx)

    def set_current_image(self, text):
        self.current_filename = text
        print(f"You selected {text}")


    def get_extrinsic_matrix(self):
        print('Current:' , self.current_filename)
        if self.current_filename:
            img = cv2.imread(self.current_filename)
            h,  w = img.shape[:2]
            if self.instrinc_matrix is not None and self.dist is not None:
                extrinsic, roi = cv2.getOptimalNewCameraMatrix(self.instrinc_matrix, self.dist, (w,h), 1, (w,h))
                print(extrinsic)
            else:
                print("Your haven't done 1.1 or 1.2")

    def get_distortion_matrix(self):
        print(self.dist)


# class Quiz2():
#     def backgroundSubtraction(self):
#         cap = cv2.VideoCapture('src/bgSub.mp4')
#         fgbg = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000)

#         while(cap.isOpened()):
#             ret, frame = cap.read()

#             try:
#                 subframe = fgbg.apply(frame)
#                 cv2.imshow('Original', frame)
#                 cv2.moveWindow('Original',250,200)
#                 cv2.imshow('Subtracted', subframe)
#                 cv2.moveWindow('Subtracted',600,200)
#             except:
#                 break

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break


# class Quiz4():
#     def __init__(self):
#         pass

#     def projection(self):
#         intrinsic = np.array(
#             [[2225.49585482, 0, 1025.5459589],
#             [0, 2225.18414074, 1038.58518846],
#             [0, 0, 1]])

#         distortion = np.array(
#             [[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]])

#         params = []

#         bmp1 = np.array(
#             [[-0.97157425, -0.01827487, 0.23602862, 6.81253889],
#             [0.07148055, -0.97312723, 0.2188925, 3.37330384],
#             [0.22568565, 0.22954177, 0.94677165, 16.71572319]])
#         params.append(bmp1)

#         bmp2 = np.array(
#             [[-0.8884799, -0.14530922, -0.435303, 3.3925504],
#             [0.07148066, -0.98078915, 0.18150248, 4.36149229],
#             [-0.45331444, 0.13014556, 0.88179825, 22.15957429]])
#         params.append(bmp2)

#         bmp3 = np.array(
#             [[-0.52390938, 0.22312793, 0.82202974, 2.86774801],
#             [0.00530458, -0.96420621, 0.26510046, 4.70990021],
#             [0.85175747, 0.14324914, 0.50397308, 12.98147662]])
#         params.append(bmp3)

#         bmp4 = np.array(
#             [[-0.63108673, 0.53013053, 0.566296, 1.22781875],
#             [0.13263301, -0.64553994, 0.75212145, 3.48023006],
#             [0.76428923, 0.54976341, 0.33707888, 10.9840538]])
#         params.append(bmp4)

#         bmp5 = np.array(
#             [[-0.87676843, -0.23020567, 0.42223508, 4.43641198],
#             [0.19708207, -0.97286949, -0.12117596, 0.67177428],
#             [0.43867502, -0.02302829, 0.89835067, 16.24069227]])
#         params.append(bmp5)

#         pyramid = np.array(
#             [[3,3,-4],[1,1,0],[1,5,0],[5,5,0],[5,1,0]]).astype(np.float32)

#         # red color
#         red  = (0,0,255)

#         for i in range(5):
#             # rvec and tvec
#             rotation_v = params[i][:,:3]
#             translation_v = params[i][:,3]

#             # 3d to 2d mapping points
#             pts, _ = cv2.projectPoints(pyramid, rotation_v, translation_v, intrinsic, distortion)

#             # input image
#             img = cv2.imread('src/{}.bmp'.format(i+1))

#             # base square
#             img = cv2.line(img, (pts[1][0][0],pts[1][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
#             img = cv2.line(img, (pts[2][0][0],pts[2][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)
#             img = cv2.line(img, (pts[3][0][0],pts[3][0][1]),(pts[4][0][0],pts[4][0][1]), red, 10)
#             img = cv2.line(img, (pts[4][0][0],pts[4][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)

#             # side lines to vertex
#             img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)
#             img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
#             img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)
#             img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[4][0][0],pts[4][0][1]), red, 10)

#             # resize to view
#             resize = cv2.resize(img, (600,600))
#             cv2.imshow('projection', resize)
#             time.sleep(0.5)
#             cv2.waitKey(30)

#         cv2.waitKey(0)

class Quiz3():
    def __init__(self):
        self.imageL = cv2.imread('src/Q3_Image/imL.png', 0)
        self.imageR = cv2.imread('src/Q3_Image/imR.png', 0)

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
        disparity = stereo.compute(self.imageL, self.imageR)
        plt.imshow(disparity, 'gray')
        plt.show()
