from PyQt5.QtWidgets import QMessageBox
# Image processing use case
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.ticker as ticker
import numpy as np

# Training use case
import tensorflow as tf
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

from lenet import LeNet

class Quiz1():
    def __init__(self):
        self.image = cv2.imread('images/dog.bmp')
        self.image_color = cv2.imread('images/color.png')

    def showImage(self):
        height, width, channels = self.image.shape
        cv2.imshow('Dog', self.image)
        print("Height =", height)
        print("Width =", width)

    def colorConversion(self):
        b, g, r = cv2.split(self.image_color)
        converted_image = cv2.merge((g, r, b))
        cv2.imshow('Color', self.image_color)
        cv2.imshow('Color BGR to RGB', converted_image)

    def imageFlipping(self):
        flipped_image = cv2.flip(self.image, 1)
        cv2.imshow('Dog flipped', flipped_image)

    def imageBlending(self):
        flipped_image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_BGR2RGB)
        normal_image =cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        blended_image = cv2.addWeighted(flipped_image, 0.5, normal_image, 0.5, 0)
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)
        im = plt.imshow(blended_image)
        ax.margins(x=0)

        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

        blend_slider = Slider(ax_slider, 'Blend', 0.0, 1.0, valinit=0.5, valstep=0.01)


        def update(val):
            blend_slider.val
            im.set_data(cv2.addWeighted(flipped_image, val, normal_image, 1- val, 0))
            fig.canvas.draw_idle()

        blend_slider.on_changed(update)

        plt.show()

class Quiz2():
    def __init__(self):
        self.image = cv2.imread('images/QR.png')

    def globalThreshold(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('QR code global', thresh)

    def localThreshold(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('QR code local', thresh)

class Quiz3():
    def __init__(self, angle, scale, tx, ty):
        self.image = cv2.imread('images/OriginalTransform.png')
        self.image_p = cv2.imread('images/OriginalPerspective.png')
        self.angle = angle
        self.scale = scale
        self.tx = tx
        self.ty = ty
        self.counter = 0

    def imageTransforms(self):
        cv2.imshow('Box', self.image)

        height, width, channels = self.image.shape
        M = cv2.getRotationMatrix2D((130, 125), float(self.angle.text()), float(self.scale.text()))
        rotated_and_scaled = cv2.warpAffine(self.image, M, (width, height))
        M = np.float32([[1, 0, int(self.tx.text())], [0, 1, int(self.ty.text())]])
        translated = cv2.warpAffine(rotated_and_scaled, M, (width, height))

        plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
        plt.show()

    def imagePerspective(self):
        self.editing_image = self.image_p.copy()
        cv2.imshow('original', self.image_p)
        cv2.moveWindow('original', 50, 50)
        self.pts = []
        cv2.setMouseCallback('original', self.get_points)


    def get_points(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.editing_image = cv2.circle(self.editing_image, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow('original', self.editing_image)
            self.pts.append([x, y])
            self.counter += 1
            if self.counter == 4:
                cv2.destroyWindow('original')
                print(self.pts)
                pts_target = np.float32(self.pts)
                pts_frame = np.float32([[20,20], [450,20], [450,450], [20,450]])
                M = cv2.getPerspectiveTransform(pts_target,pts_frame)
                dst = cv2.warpPerspective(self.image_p,M,(430,430))
                cv2.imshow('perspective', dst)
                self.counter = 0

class Quiz4():
    def __init__(self):
        self.image = cv2.imread('images/School.jpg')
        self.gray = self.custom_grayscale(self.image)
    
    def gaussianSmooth(self):
        image = self.image        
        cv2.imshow('gray', self.gray)

        xvalues = np.linspace(-1, 1, 3)
        yvalues = np.linspace(-1, 1, 3)
        xx, yy = np.meshgrid(xvalues, yvalues)

        kernel = np.zeros((3,3))
        sigma = 1
        for i in range(3):
            for j in range(3):
                x1 = 1 / (2 * np.pi * (sigma ** 2))
                x2 = np.exp(-(xx[i][j] ** 2 + yy[i][j] ** 2) / (2 * (sigma ** 2)))
                value = x1 * x2
                kernel[i][j] = value

        gaussian_smooth = self.apply_filter(self.gray, kernel).astype(np.uint8)
        cv2.imshow('Gaussian Smooth', gaussian_smooth)

    def sobel_x(self):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_x = self.apply_sobel(self.gray, kernel).astype(np.uint8)
        cv2.imshow('Sobel X', sobel_x)

    def sobel_y(self):
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sobel_y = self.apply_sobel(self.gray, kernel).astype(np.uint8)
        cv2.imshow('Sobel Y', sobel_y)

    def magnitude(self):
        xkernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ykernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sobel_x = self.apply_sobel(self.gray, xkernel).astype(np.uint8)
        sobel_y = self.apply_sobel(self.gray, ykernel).astype(np.uint8)

        magnitude = np.zeros(sobel_x.shape)
        for i in range(sobel_x.shape[0]):
            for j in range(sobel_x.shape[1]):
                magnitude[i][j] = np.sqrt(sobel_x[i][j] ** 2 + sobel_y[i][j] ** 2)
            
        magnitude = magnitude.astype(np.uint8)

        cv2.imshow('Magnitude', magnitude)

    def apply_filter(self, image, kernel):
        image_pad = np.pad(image, ((1,1), (1,1)), mode='constant', constant_values=0).astype(np.float32)

        height, width = image_pad.shape
        image_conv = np.zeros(image_pad.shape)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                crop = image_pad[i-1:i-1+3, j-1:j-1+3]
                x = crop.flatten() * kernel.flatten()
                image_conv[i][j] = abs(x.sum())

        return image_conv[1:-1, 1:-1]

    def apply_sobel(self, image, kernel):
        image_pad = np.pad(image, ((1,1), (1,1)), mode='constant', constant_values=0).astype(np.float32)

        height, width = image_pad.shape
        image_conv = np.zeros(image_pad.shape)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                crop = image_pad[i-1:i-1+3, j-1:j-1+3]
                x = crop.flatten() * kernel.flatten()
                image_conv[i][j] = abs(x.sum())

        filtered = image_conv[1:-1, 1:-1]
        min_value = np.amin(filtered)
        max_value = np.amax(filtered)
        the_range = max_value - min_value

        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                filtered[i][j] = (filtered[i][j] - min_value) / the_range * 255 

        return filtered

    def custom_grayscale(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]

        # to gray scale
        gray = b / 3 + g / 3 + r /3
        gray = gray.astype(np.uint8)
        return gray

class GetIterationLosses(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Quiz5():
    def __init__(self):
        self.minist_data = tf.keras.datasets.mnist.load_data()
        self.batch_size = 32
        self.learn_rate = 0.001
        self.optimizer = 'SGD'
        self.test_number = None

    def showTrainImage(self):
        (x_train, y_train), _ = self.minist_data
        indexes = random.sample(range(len(y_train)), 10)

        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            ax.imshow(x_train[indexes[i]])
            ax.set_title(y_train[indexes[i]])
        
        plt.show()

    def showHyperparemeters(self):
        print('hyperparameters:')
        print('batch size:', self.batch_size)
        print('learning rate:', self.learn_rate)
        print('optimizer:', self.optimizer)

    def transformDataset(self):
        (x_train, y_train), (x_test, y_test) = self.minist_data
        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
            x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
        else:
            x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
            x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        # scale to [0, 1.0]
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def train_1_epoch(self):
        (x_train, y_train), (x_test, y_test) = self.transformDataset()

        print("[INFO] compiling model...")
        optimizer = SGD(lr=self.learn_rate)
        model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        print("[INFO] training...")

        custom_callback = GetIterationLosses()
        history = model.fit(x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=1,
            verbose=1,
            callbacks=[custom_callback]
        )
        # show the accuracy on the testing set
        print("[INFO] evaluating...")
        (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

        plt.plot(custom_callback.losses)
        plt.suptitle('1st Epoch Losses')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()

    def train_model(self):
        (x_train, y_train), (x_test, y_test) = self.transformDataset()

        print("[INFO] compiling model...")
        optimizer = SGD(lr=self.learn_rate)
        model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        print("[INFO] training...")

        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=50, verbose=1)
        
        # show the accuracy on the testing set
        print("[INFO] evaluating...")
        (loss, accuracy) = model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

        model.save("inference_output/train_50_epochs.hdf5", overwrite=True)

        acc_ax = plt.subplot(121)
        acc_ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        acc_ax.plot(history.history['accuracy'])
        acc_ax.set_title('Accuracy')
        acc_ax.set_ylabel('%')
        acc_ax.set_xlabel('Epoch')

        loss_ax = plt.subplot(122)
        loss_ax.plot(history.history['loss'])
        loss_ax.set_title('Loss')
        loss_ax.set_ylabel('Loss')
        loss_ax.set_xlabel('Epoch')

        plt.show()

    def predit_number(self):
        test_number_index = int(self.test_number.text())
        _, (testData, _) = self.minist_data
        _, (x_test, _) = self.transformDataset()

        target_image = testData[test_number_index]
        zoomed_image = cv2.resize(target_image, (200, 200))
        cv2.imshow('Chosen number (zoomed)', zoomed_image)

        model = None
        try:
            model = load_model('inference_output/train_50_epochs.hdf5')
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Can't load model / model missing / run 5.4")
            retval = msg.exec_()

        if model:
            probs = model.predict(x_test[np.newaxis, test_number_index])
            answer = probs.argmax(axis=1)
            plt.bar(range(10), probs[0])
            print(np.asscalar(np.array(answer)))
            plt.title('The prodicted number is {}'.format(answer[0]))
            plt.xlabel('Number')
            plt.ylabel('Probability (%)')
            plt.show()