import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*8,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('src/Q1_Image/*.bmp')


axes=[]
fig=plt.figure()

for i, fname in enumerate(images):
    img = cv.imread(fname)

    height, width, channels = img.shape

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (11,8), None)
    # If found, add object points, image points (after refining them)
    # print(ret, corners)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
#         cv.drawChessboardCorners(img, (11,8), corners2, ret)

#         img = cv.resize(img, (1024, 1024))                    # Resize image
#         cv.imshow(fname, img)
#         cv.waitKey(500)
# cv.destroyAllWindows()

    #     axes.append(fig.add_subplot(4 , 4, i+1))
    #     plt.imshow(img)
    # else:
    #     axes.append(fig.add_subplot(4 , 4, i+1))
    #     plt.imshow(img)

# plt.show()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


