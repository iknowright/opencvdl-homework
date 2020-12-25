from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import random
import glob
import cv2

files = glob.glob("src/Q5_Image/*.jpg")

# 從參數讀取圖檔路徑
image_file = random.choice(files)

# 載入訓練好的模型
net = load_model('model-resnet50-final.h5')

cls_list = ['cats', 'dogs']

img = image.load_img(image_file, target_size=(224, 224))
if img is not None:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    print(image_file)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))

    image = cv2.imread(image_file)
    cv2.imshow(f'{pred[top_inds[0]]:.3f}  {cls_list[top_inds[0]]}', image)
    cv2.waitKey(0)
