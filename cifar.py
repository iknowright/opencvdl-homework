import pickle
import random

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


import numpy as np
import matplotlib.pyplot as plt

# cifar-10-python.tar.gz
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


files = [f"data_batch_{i+1}" for i in range(5)]
files.append("test_batch")

class Quiz5():
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.optimizer = 'SGD'
        self.model = None

    def plot_random(self):
        meta_data = unpickle(f"src/Q5_Data/batches.meta")
        label_names = meta_data[b'label_names']
        print(label_names)

        batch1 = unpickle(f"src/Q5_Data/{files[0]}")
        # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        print(batch1.keys())
        print(type(batch1[b'labels']))
        plt.figure(1)
        for i in range(10):
            plt.subplot(1, 10, i+1)
            print(i)
            index = random.randint(0, len(batch1[b'labels']))
            print(index)
            label = batch1[b'labels'][index]
            label_name = label_names[label].decode('ASCII')
            print(batch1[b'data'].shape)
            image_buff = batch1[b'data'][index].reshape(3, 32, 32)
            image_buff = np.swapaxes(image_buff, 0, 2)
            image_buff = np.swapaxes(image_buff, 0, 1)
            print(image_buff.shape)
            plt.imshow(image_buff)
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
        self.batch_size = 32
        self.learning_rate = 0.001
        self.optimizer = 'SGD'
        print("Hyperparameters:")
        print(f"batch_size: {batch_size}")
        print(f"learning_rate: {learning_rate}")
        print(f"optimizer: {optimizer}")

    def get_model_data(self):
        # generate testing data
        data = unpickle(f"src/Q5_Data/test_batch")
        image_data = data[b'data'][:2000]
        print(image_data.shape)
        image = image_data.reshape(2000, 3, 32, 32)
        image = np.swapaxes(image, 1, 3)
        image = np.swapaxes(image, 1, 2)
        y_test = data[b'labels'][:2000]
        x_test = image
        y_test = utils.to_categorical(y_test, 10, dtype=int)

        # data_batch_1
        # generate training data
        data = unpickle(f"src/Q5_Data/data_batch_1")
        image_data = data[b'data']
        print(image_data.shape)
        image = image_data.reshape(10000, 3, 32, 32)
        image = np.swapaxes(image, 1, 3)
        image = np.swapaxes(image, 1, 2)
        y_train = data[b'labels']
        x_train = image
        y_train = utils.to_categorical(y_train, 10, dtype=int)

        return x_train, y_train, x_test, y_test


    def train(self):
        x_train, y_train, x_test, y_test = self.get_model_data()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        opt = SGD(learning_rate=0.1)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt)
        history = self.model.fit(x_train, y_train,
            batch_size=self.batch_size,
            epochs=20,
            verbose=1,
            validation_data=(x_test, y_test)
        )
        with open('training_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        self.model.save('trained_model.h5')

    # def predit(self):
    #     with open('training_history', 'rb') as fo:
    #         data = pickle.load(fo)
    #         print(data)


quiz = Quiz5()
# quiz.VGG16()
# quiz.train()
quiz.predit()


