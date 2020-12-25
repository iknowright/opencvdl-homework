from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import tensorflow as tf
from datetime import datetime


sess = tf.InteractiveSession()

# 資料路徑
DATASET_PATH  = 'sample'

# 影像大小
IMAGE_SIZE = (224, 224)

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 8

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 20

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-final.h5'


# 透過 data augmentation 產生訓練與驗證用的影像資料
train_data_generator = ImageDataGenerator()

train_batches = train_data_generator.flow_from_directory(
    './PetImages',
    target_size=IMAGE_SIZE,
    interpolation='bicubic',
    class_mode='categorical',
    shuffle=True,
    batch_size=BATCH_SIZE,
    subset='training'
)

valid_batches = train_data_generator.flow_from_directory(
    './PetImages',
    target_size=IMAGE_SIZE,
    interpolation='bicubic',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
# print(net_final.summary())

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# 訓練模型
net_final.fit_generator(train_batches,
    steps_per_epoch = train_batches.samples // BATCH_SIZE,
    validation_data = valid_batches,
    validation_steps = valid_batches.samples // BATCH_SIZE,
    epochs = NUM_EPOCHS,
    callbacks=[tensorboard_callback]
)

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)