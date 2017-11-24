import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

def add(img, a):

	# Define the scaling function.
	f = lambda x: p(a, x)

	# Get image dimensions.
	height = img.shape[0]
	width = img.shape[1]

	# Create vector of radii powers [r, r**2, ..., r**n] for each pixel.
	x, y = np.meshgrid(np.arange(width) - width/2, np.arange(height) - height/2)
	r = np.sqrt(x ** 2 + y ** 2) / np.sqrt((width/2) ** 2 + (height/2) ** 2)

	s = f(r)
	J = np.zeros(img.shape, np.float32)
	J[:,:,0] = s * img[:,:,0]
	J[:,:,1] = s * img[:,:,1]
	J[:,:,2] = s * img[:,:,2]

	return J

def remove(img, W, b):

	# Define the scaling function.
	f = lambda x: p(np.concatenate([b, W.flatten()]), x)

	# Get image dimensions.
	height = img.shape[0]
	width = img.shape[1]

	# Create vector of radii powers [r, r**2, ..., r**n] for each pixel.
	x, y = np.meshgrid(np.arange(width) - width/2, np.arange(height) - height/2)
	r = np.sqrt(x ** 2 + y ** 2) / np.sqrt((width/2) ** 2 + (height/2) ** 2)

	s = 1/f(r)
	J = np.zeros(img.shape, np.float32)
	J[:,:,0] = np.clip(s * img[:,:,0], 0, 255)
	J[:,:,1] = np.clip(s * img[:,:,1], 0, 255)
	J[:,:,2] = np.clip(s * img[:,:,2], 0, 255)

	return J

# Get polynomial from coefficient vector a.
def p(a, x):

	y = np.zeros(x.shape)
	for i in range(a.shape[0]):
		y += a[i] * x ** i

	return y

def poly(W, b, x):
	return p(np.concatenate([b, W.flatten()]), x)