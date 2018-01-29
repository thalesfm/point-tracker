import cv2
import numpy as np

import pygame
from pygame.locals import *

import random

from detect import detect_mixed as detect
from solvepnp import solvepnp_python as solvepnp
import graphics


min_dist = 10
num_guess = 5

objp = np.array([[ 0.   , 0.   , 0.   ],
                 [ 0.041, 0.   , 0.   ],
                 [ 0.   , 0.052, 0.   ],
                 [ 0.041, 0.052, 0.   ]])
objp[:,2] =-objp[:,0]
objp[:,0] = objp[:,1]
objp[:,1] = 0.


'''
def order(keyp):
	assert len(keyp) >= len(objp)

	# keyp = sorted(keyp, key=lambda x: x.response)
	keyp = keyp[:4]

	imgp = [x.pt for x in keyp]
	imgp.sort(key=lambda x: -x[1])
	imgp[:2] = sorted(imgp[:2], key=lambda x: x[0])
	imgp[2:] = sorted(imgp[2:], key=lambda x: x[0])

	return np.array(imgp)
'''


def order(keyp, rvec, tvec, cmat, dist):
	imgp_last, _ = cv2.projectPoints(objp, rvec, tvec, cmat, dist)
	imgp = []

	for p in imgp_last:
		i = np.linalg.norm(keyp - p, axis=1).argmin()
		q = keyp[i]
		imgp.append(q)

	return np.array(imgp)


def valid(imgp):
	for i, p in enumerate(imgp):
		for q in imgp[i+1:]:
			if np.linalg.norm(p - q) < min_dist:
				return False

	return True


def track(img, cmat, dist, rvec, tvec, flag):
	keyp = detect(img)

	for p in keyp:
		p = tuple(p.astype(np.int32))
		img = cv2.drawMarker(img, p, (255, 255, 0))

	if len(keyp) < len(objp):
		return False, img, rvec, tvec

	imgp = []

	if flag:
		p = order(keyp, rvec, tvec, cmat, dist)
		imgp.append(p)

	for _ in range(num_guess):
		i = np.random.choice(len(keyp), len(objp))
		p = keyp[i]
		imgp.append(p)

	imgp = [p for p in imgp if valid(p)]

	for p in imgp:
		ret, rv, tv = solvepnp(objp, p, cmat, dist, rvec, tvec, flag)

		if ret:
			return True, img, rv, tv

	return False, img, rvec, tvec


def axis(img, rvec, tvec, cmat, dist, color=255):
	axisp = 5e-2 * np.eye(4, 3).astype(np.float32)

	imgp, _ = cv2.projectPoints(axisp, rvec, tvec, cmat, dist)
	imgp = [tuple(x) for x in imgp.reshape((-1,2))]

	try:
		img = cv2.line(img, imgp[3], imgp[0], (0, 0, color), 2)
		img = cv2.line(img, imgp[3], imgp[1], (0, color, 0), 2)
		img = cv2.line(img, imgp[3], imgp[2], (color, 0, 0), 2)
	except:
		pass

	return img


def main():
	cap = cv2.VideoCapture(0)
	assert cap.isOpened(), 'failed to open capture'

	_, img = cap.read()
	height, width = img.shape[:2]

	pygame.init()
	pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

	graphics.init()

	with np.load('calib.npz') as data:
		cmat, dist = data['mat'], data['dist']

	flag = False
	rvec = np.zeros(3)
	tvec = np.zeros(3)

	while True:
		_, img = cap.read()

		flag, img, rvec, tvec = track(img, cmat, dist, rvec, tvec, flag)

		color = 255 if flag else 100
		# img = axis(img, rvec, tvec, cmat, dist, color)

		graphics.clear()
		graphics.imshow(img)

		if flag:
			graphics.drawteapot(rvec, tvec, cmat, width, height)

		pygame.display.flip()
		pygame.time.wait(17)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				cap.release()
				pygame.quit()
				quit()


if __name__ == '__main__':
	main()
