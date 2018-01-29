import cv2
import numpy as np
import pyassimp as assimp

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

import time


fname = 'data/teapot.obj'
scale = 0.1

vertex_shader_source = """
#version 120

uniform mat4 mvpMatrix;
uniform mat3 normalMatrix;

attribute vec3 vertPosition;
attribute vec3 vertNormal;

varying vec3 fragNormal;

void main()
{
	gl_Position = mvpMatrix * vec4(vertPosition, 1.0f);
	fragNormal = normalMatrix * vertNormal;
}
"""

fragment_shader_source = """
#version 120

vec3 ambientColor = vec3(0.3f, 0.3f, 0.3f);
vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
vec3 lightDir = normalize(vec3(1.0f, 1.0f, 1.0f));

varying vec3 fragNormal;

void main()
{
	float intensity = max(0.0f, dot(fragNormal, lightDir));
	vec3 color = ambientColor + lightColor * intensity;
	gl_FragColor = vec4(color, 1.0f);
}
"""

def init_program():
	global program

	program = compileProgram(
		compileShader(vertex_shader_source, GL_VERTEX_SHADER),
		compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
	)


def init_buffer(target, data):
	buffer = glGenBuffers(1)
	glBindBuffer(target, buffer)
	glBufferData(target, data, GL_STATIC_DRAW)
	glBindBuffer(target, 0)

	return buffer


def init_all_buffers():
	global vbuffer, nbuffer, ibuffer, count

	scene = assimp.load(fname)
	mesh = scene.meshes[0]
	assimp.release(scene)

	vbuffer = init_buffer(GL_ARRAY_BUFFER, scale * mesh.vertices)
	nbuffer = init_buffer(GL_ARRAY_BUFFER, mesh.normals)
	ibuffer = init_buffer(GL_ELEMENT_ARRAY_BUFFER, mesh.faces)

	count = len(mesh.faces.flatten())

def init_texture():
	global texture
	texture = glGenTextures(1)

	glBindTexture(GL_TEXTURE_2D, texture)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)


def init():
	init_program()
	init_all_buffers()
	init_texture()

	glEnable(GL_CULL_FACE)
	glEnable(GL_DEPTH_TEST)


def clear():
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def imshow(img):
	img = np.flip(img, axis=2)

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0],
		0, GL_RGB, GL_UNSIGNED_BYTE, img
	)

	glUseProgram(0)
	glEnable(GL_TEXTURE_2D)
	glLoadIdentity()

	glBegin(GL_QUADS)

	glTexCoord( 0., 1.)
	glVertex2f(-1.,-1.)
	glTexCoord( 1., 1.)
	glVertex2f( 1.,-1.)
	glTexCoord( 1., 0.)
	glVertex2f( 1., 1.)
	glTexCoord( 0., 0.)
	glVertex2f(-1., 1.)

	glEnd()


def modelview_matrix(rvec, tvec):
	rvec = np.array([ 1.,-1.,-1.]) * rvec
	tvec = np.array([ 1.,-1.,-1.]) * tvec

	mat = np.eye(4)
	mat[:3,:3] = cv2.Rodrigues(rvec)[0]
	mat[:3, 3] = tvec

	return mat


def projection_matrix(cmat, width, height):
	fx = 2 * cmat[0,0] / width
	fy = 2 * cmat[1,1] / height

	mat = np.array([[ fx, 0., 0., 0.],
	                [ 0., fy, 0., 0.],
	                [ 0., 0., 1., 0.],
	                [ 0., 0.,-1., 0.]])

	return mat


def drawteapot(rvec, tvec, cmat, width, height):
	mvmat = modelview_matrix(rvec, tvec)
	pmat = projection_matrix(cmat, width, height)
	mvpmat = pmat.dot(mvmat)
	nmat = np.linalg.inv(mvmat[:3,:3]).T

	glUseProgram(program)

	glUniformMatrix4fv(0, 1, True, mvpmat)

	glUniformMatrix3fv(1, 1, True, nmat)

	glEnableVertexAttribArray(0)
	glBindBuffer(GL_ARRAY_BUFFER, vbuffer)
	glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

	glEnableVertexAttribArray(1)
	glBindBuffer(GL_ARRAY_BUFFER, nbuffer)
	glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, None)

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer)

	glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glDisableVertexAttribArray(1)
	glDisableVertexAttribArray(0)
	glUseProgram(0)


def main():
	pygame.init()
	pygame.display.set_mode((480, 360), DOUBLEBUF | OPENGL)
	init()

	while True:
		rvec = time.clock() * np.array([ 0., 1., 0.])
		tvec = np.array([ 0., 0.03, 0.3])
		cmat = np.diag([1., 1.56, 1.])

		clear()
		drawteapot(rvec, tvec, cmat, 1., 1.)
		pygame.display.flip()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()


if __name__ == '__main__':
    main()

