import cv2
import numpy as np


min_area = 0
max_area = 300
threshold = 254


params = cv2.SimpleBlobDetector_Params()

# Configuração do detector de pontos
params.minArea = min_area
params.maxArea = max_area
params.minThreshold = 160
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255

# Criação do detector de pontos
detector = cv2.SimpleBlobDetector_create(params)


def detect_opencv(img):
	# Converte para escala de cinza
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Encontra a posição de todos os pontos
	keyp = detector.detect(gray)
	imgp = [np.array(k.pt) for k in keyp]

	return np.array(imgp)


def floodfill(img, x, y):
	center = np.zeros(2)
	area = 0
	stack = [[x, y]]

	for x, y in stack:
		if not 0 <= x < img.shape[1]:
			continue

		if not 0 <= y < img.shape[0]:
			continue

		if not img[y, x]:
			continue

		img[y, x] = False

		center += [x, y]
		area += 1

		stack.append([x, y-1])
		stack.append([x, y+1])
		stack.append([x-1, y])
		stack.append([x+1, y])

	center /= area
	return center, area


def detect_python(img):
	# Converte para escala de cinza
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Binarização da imagem
	thresh = gray >= threshold

	# Encontra todos os conjuntos de pontos contíguos
	imgp = []
	for y, x in np.ndindex(*thresh.shape):
		if thresh[y, x]:
			center, area = floodfill(thresh, x, y)

			if min_area <= area <= max_area:
				imgp.append(center)

	return np.array(imgp)


def detect_mixed(img):
	# Converte para escala de cinza
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Binarização da imagem
	thresh = (gray > threshold).astype(np.uint8)

	# Encontra todos os conjuntos de pontos contíguos
	_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	imgp = []
	for cont in contours:
		center = np.average(cont, axis=0)[0]
		area = len(cont)

		if min_area <= area <= max_area:
			imgp.append(center)

	return np.array(imgp)

