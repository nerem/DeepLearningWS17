import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import preparation
import model
import vignetting


def cross_validation_lambda(n = 4, lam_min = 1., lam_max = 3., lam_stepsize = 1., epochs = 2, k = 2, minibatch_size = 10000, ratio = 0.8, step_size = 0.1):

    steps = ((lam_max - lam_min) / lam_stepsize) + 1
    steps = (int)(steps)
    L = []
    for l in range(steps):
        lam = lam_min + l*lam_stepsize
        print(l)
        print("")
        print("Try lambda =", lam)
        print("")
        L.append(model.complete_cv(n = n, lam = lam, epochs = epochs, k = k, minibatch_size = minibatch_size, ratio = ratio, step_size = step_size))

    print("")
    print("Comparison")
    print(L)

    return L

# Cross validation trying lam_min to lam_max (included).
k = 2
lam_min = 0.1
lam_max = 0.2
lam_stepsize = 0.02
L = cross_validation_lambda(n = 2, lam_min = lam_min, lam_max = lam_max, lam_stepsize = lam_stepsize, epochs = 2, k = k, minibatch_size = 10000, ratio = 0.8, step_size = 0.1)
plt.figure(3)
x = np.linspace(lam_min, lam_max, num = ((lam_max - lam_min) / lam_stepsize) + 1)
plt.plot(x, np.array(L), 'o-')
plt.title("Validation errors for hyperparameter lambda (regularization of L_2 loss) using 2-Cross validation")
plt.xlabel("Hyperparamter: Value of lambda")
plt.ylabel("Validation error")
plt.show()