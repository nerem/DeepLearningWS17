import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import preparation
import model
import vignetting

def show(W, b):

	# Results
	print("")
	print("--- Results ---")
	print("W:", W)
	print("b:", b)

	# Show the original image.
	plt.figure(1)
	plt.subplot(1, 3, 1)
	plt.imshow(imread("cat_03.jpg"))
	plt.title("Original image")

	plt.subplot(1,3,2)
	plt.imshow(imread("cat_03_vignetted.jpg"))
	plt.title("Vignetted and disorted image")

	# Show the devignetted image.
	plt.subplot(1, 3, 3)
	plt.imshow(np.uint8(vignetting.remove(imread("cat_03_vignetted.jpg"), W, b)))
	plt.title("Disorted image with vignette removed by NN")

	# Show the function.
	x = np.linspace(0.0, 1.0, num = 100)
	plt.figure(2)
	plt.plot(x , 1/vignetting.poly(W, b, x))
	plt.title("1/p(r)")
	plt.show()

	return


# Training for fixed n = 2 and lambda = 0 (no regularization)
W, b = model.complete(n = 1, lam = 0, epochs = 1, minibatch_size = 1000, ratio = 0.8, step_size = 0.1)
show(W, b)

# W, b = model.complete(n = 3, lam = 0, epochs = 5, minibatch_size = 10000, ratio = 0.8, step_size = 0.1)
# show(W, b)
