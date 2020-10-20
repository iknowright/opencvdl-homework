import pickle
import random

from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


import numpy as np
import matplotlib.pyplot as plt

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

    def train(self):
        x_train, y_train, x_test, y_test = self.get_model_data()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        y_train_cat = utils.to_categorical(y_train, 10)
        y_test_cat = utils.to_categorical(y_test, 10)
        opt = SGD(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = self.model.fit(x_train, y_train,
            batch_size=self.batch_size,
            epochs=100,
            verbose=1,
            validation_data=(x_test, y_test),
        )
        with open('training_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        self.model.save('trained_model.h5')

    def predit(self):
        with open('training_history', 'rb') as fo:
            data = pickle.load(fo)
            print(data)
            print(len(data["loss"]))




quiz = Quiz5()
# quiz.VGG16()
# quiz.train()
# quiz.predit()

