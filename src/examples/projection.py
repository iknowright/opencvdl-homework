import numpy as np
import cv2
import time

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