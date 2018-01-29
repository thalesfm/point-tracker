import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('images/*')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)


    if ret:
        objpoints.append(objp)

        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)


        cv2.drawChessboardCorners(img, (7,7), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(250)

ret, mat, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    np.savez('calib.npz', mat=mat, dist=dist)

print('done.' if retval else 'failed.')

cv2.destroyAllWindows()
