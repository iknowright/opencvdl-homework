import random
import time
import pickle

from PyQt5.QtWidgets import QMessageBox
# Image processing use case
import cv2
import matplotlib.pyplot as plt
import numpy as np
# tensorflow
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


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
        self.rvecs = []
        self.tvecs = []
        self.dist = None
        self.current_filename = ''
        self.index = -1

    def chessboard_corners(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_point = np.zeros((self.cb_width * self.cb_height, 3), np.float32)
        obj_point[:,:2] = np.mgrid[0 : self.cb_width, 0 : self.cb_height].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.

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


    def get_intrinsic_and_extrinsic_matrix(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size, None, None)
        if ret:
            self.instrinc_matrix = mtx
            self.dist = dist
            self.rvecs = [cv2.Rodrigues(rvecs[i])[0] for i in range(15)]
            self.tvecs = tvecs
            print(mtx)

    def set_current_image(self, text):
        self.current_filename = text
        print(f"You selected {text}")
        self.index = self.filenames.index(text)


    def get_extrinsic_matrix(self):
        print('Current:' , self.current_filename)
        if self.index >= 0:
            print(np.concatenate((self.rvecs[self.index], self.tvecs[self.index]), axis=1))

    def get_distortion_matrix(self):
        print(self.dist)

class Quiz2():
    def __init__(self):
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane
        self.image_size = 0
        self.cb_width = 11
        self.cb_height = 8
        self.filenames = [f'src/Q2_Image/{i+1}.bmp' for i in range(5)]
        self.instrinc_matrix = None
        self.rvecs = []
        self.tvecs = []
        self.dist = None

    def chessboard_corners(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_point = np.zeros((self.cb_width * self.cb_height, 3), np.float32)
        obj_point[:,:2] = np.mgrid[0 : self.cb_width, 0 : self.cb_height].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.

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
        self.image_size = gray.shape[::-1]

    def get_intrinsic_and_extrinsic_matrix(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size, None, None)
        if ret:
            self.instrinc_matrix = mtx
            self.dist = dist
            self.rvecs = [cv2.Rodrigues(rvecs[i])[0] for i in range(5)]
            self.tvecs = tvecs
            print(mtx)

    def projection(self):
        self.chessboard_corners()
        self.get_intrinsic_and_extrinsic_matrix()

        # red color
        red  = (0,0,255)

        pyramid = np.array(
            [[3,3,-3],[1,1,0],[3,5,0],[5,1,0]]).astype(np.float32)

        for i in range(5):
            print(i, i+1)
            # rvec and tvec
            rotation_v = self.rvecs[i]
            translation_v = self.tvecs[i]

            # 3d to 2d mapping points
            pts, _ = cv2.projectPoints(pyramid, rotation_v, translation_v, self.instrinc_matrix, self.dist)

            # input image
            img = cv2.imread(self.filenames[i])

            # base triangle
            img = cv2.line(img, (pts[1][0][0],pts[1][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
            img = cv2.line(img, (pts[2][0][0],pts[2][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)
            img = cv2.line(img, (pts[3][0][0],pts[3][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)

            # side lines to vertex
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[1][0][0],pts[1][0][1]), red, 10)
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[2][0][0],pts[2][0][1]), red, 10)
            img = cv2.line(img, (pts[0][0][0],pts[0][0][1]),(pts[3][0][0],pts[3][0][1]), red, 10)

            # resize to view
            resize = cv2.resize(img, (1024,1024))
            cv2.imshow(f'projection-{self.filenames[i]}', resize)
            cv2.waitKey(2000)

    cv2.destroyAllWindows()


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


class Quiz4():
    def __init__(self):
        self.imageL = cv2.imread('src/Q4_Image/Aerial1.jpg')
        self.img1 = cv2.imread('src/Q4_Image/Aerial1.jpg', 0)
        self.imageR = cv2.imread('src/Q4_Image/Aerial2.jpg')
        self.img2 = cv2.imread('src/Q4_Image/Aerial2.jpg', 0)
        self.kp1 = []
        self.kp2 = []
        self.des1 = []
        self.des2 = []

    def sift(self):
        sift = cv2.xfeatures2d.SIFT_create()

        gray1= cv2.cvtColor(self.imageL, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        size1 = [kp.size for kp in kp1]
        # self.kp1, self.des1 = kp1, des1
        tuple_list = zip(kp1, des1, size1)
        # self.kp1, self.des1 = kp1, des1
        tuple_list = sorted(tuple_list, key=lambda t: t[2], reverse=True)[:7]
        self.kp1 = [t[0] for t in tuple_list]
        self.des1 = [t[1] for t in tuple_list]
        img = cv2.drawKeypoints(gray1, self.kp1, self.imageL)
        # cv2.imshow('gray1', img)
        cv2.imwrite('FeatureAerial1.jpg', img)

        gray2= cv2.cvtColor(self.imageR, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        size2 = [kp.size for kp in kp2]
        # self.kp1, self.des1 = kp1, des1
        tuple_list = zip(kp2, des2, size2)
        # self.kp1, self.des1 = kp1, des1
        tuple_list = sorted(tuple_list, key=lambda t: t[2], reverse=True)[:7]
        self.kp2 = [t[0] for t in tuple_list]
        self.des2 = [t[1] for t in tuple_list]
        img = cv2.drawKeypoints(gray2, self.kp2, self.imageR)
        # cv2.imshow('gray2', img)
        cv2.imwrite('FeatureAerial2.jpg', img)


    def matcher(self):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(np.array(self.des1), np.array(self.des2), k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(self.imageL,self.kp1,self.imageR, self.kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()


class Quiz5():
    def __init__(self):
        self.batch_size = 256
        self.learning_rate = 0.1
        self.optimizer = 'SGD'
        self.model = None

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def get_model_data(self):
        with open('src/Q5_Data/cifar10.data', 'rb') as fo:
            data = pickle.load(fo)
            (x_train, y_train), (x_test, y_test) = data
            return x_train, y_train, x_test, y_test

    def plot_random(self):
        meta_data = self.unpickle(f"src/Q5_Data/batches.meta")
        label_names = meta_data[b'label_names']
        print(label_names)

        x_train, y_train, _, _ = self.get_model_data()

        print(y_train.shape)
        indices = random.sample(range(50000), k=10)
        plt.figure(1)
        for i, index in enumerate(indices):
            plt.subplot(1, 10, i+1)
            label = y_train[index][0]
            label_name = label_names[label].decode('ASCII')
            plt.imshow(x_train[index])
            plt.title(label_name)

        plt.show()

    def VGG16(self):
        input_shape = (32, 32, 3)

        model = Sequential([
            Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
                activation='relu'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            Conv2D(256, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            Conv2D(512, (3, 3), activation='relu', padding='same',),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        self.model = model

    def parameter(self):
        print("Hyperparameters:")
        print(f"batch_size: {self.batch_size}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"optimizer: {self.optimizer}")
