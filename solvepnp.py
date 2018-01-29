import cv2
import numpy as np


rtol = 3e-3
damping = 1e3


default_rvec = np.array([ 0., 0., 0.])
default_tvec = np.array([ 0., 0., 1.])


def solvepnp_opencv(objp, imgp, cmat, dist, rvec, tvec, flag):
	return cv2.solvePnP(objp, imgp, cmat, dist, rvec, tvec, flag)


def solvepnp_python(objp, imgp, cmat, dist, rvec, tvec, flag):
	imgp = imgp.flatten()

	# Constrói o vetor posição a partir dos vetores de rotação e translação
	if flag:
		vecs = np.concatenate((rvec, tvec))
	# Caso a solução anterior não esteja disponível, escolhe valores padrão
	else:
		vecs = np.concatenate((default_rvec, default_tvec))

	# Limite máximo de 30 iterações
	for i in range(30):
		rvec, tvec = vecs[:3], vecs[3:]

		# Projeta a estimativa e calcula a jacobiana
		p, jac = cv2.projectPoints(objp, rvec, tvec, cmat, dist)
		p, jac = p.flatten(), jac[:,:6]

		# Retorna caso o erro seja menor que a tolerância desejada
		rloss = np.linalg.norm(imgp - p) / np.linalg.norm(imgp)
		if rloss < rtol:
			print(i+1, 'iters')
			return True, rvec, tvec

		a = jac.T.dot(jac) + damping * np.eye(6)
		b = jac.T.dot(imgp - p)

		# Realiza uma iteração do método
		try:
			dvec = np.linalg.solve(a, b)
			vecs += dvec
		# Em caso de sistema singular escolhe outro chute incial
		except:
			rvec[:] = default_rvec + np.pi * np.random.rand(3)
			tvec[:] = default_tvec + np.random.randn(3)

	return False, rvec, tvec

