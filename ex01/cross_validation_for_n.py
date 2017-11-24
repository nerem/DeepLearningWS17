import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import preparation
import model
import vignetting


def cross_validation(n_min = 1, n_max = 2, lam = 0, epochs = 2, k = 2, minibatch_size = 10000, ratio = 0.8, step_size = 0.1):

	L = []
	for n in range(n_min, n_max + 1):
		print("")
		print("Try n =", n)
		print("")
		L.append(model.complete_cv(n = n, lam = lam, epochs = epochs, k = k, minibatch_size = minibatch_size, ratio = ratio, step_size = step_size))

	print("")
	print("Comparison")
	print(L)

	return L

# Cross validation trying n_min to n_max (included).
k = 5
n_min = 0
n_max = 4
L = cross_validation(n_min = n_min, n_max = n_max, lam = 0, epochs = 10, k = k, minibatch_size = 10000, ratio = 0.8, step_size = 0.1)
plt.figure(3)
x = np.linspace(n_min, n_max, num = n_max - n_min + 1)
plt.plot(x, np.array(L), 'o-')
plt.title("Validation errors for hyperparameter n using 5-Cross validation")
plt.xlabel("Hyperparamter: degree of polynomial n")
plt.ylabel("Validation error")
plt.show()