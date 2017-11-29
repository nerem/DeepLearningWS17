import numpy as np
import tensorflow as tf
import h5py


numpy_input_fn = tf.estimator.inputs.numpy_input_fn

def get_train_input_fn(batch_size, shuffle = True, one_hot = True, scaled = True):
    x, y = load(one_hot = one_hot, scaled = scaled)
    return numpy_input_fn(x = {"x" : x}, y = y, batch_size = batch_size, num_epochs = None, shuffle = shuffle)

def get_test_input_fn(one_hot = True, scaled = True):
    x, y = load(subset = "test", one_hot = one_hot, scaled = scaled)
    return numpy_input_fn(x = {"x" : x}, y = y, num_epochs = 1, shuffle = False)

def load(subset = "train", one_hot = False, scaled = False):
    with  h5py.File("MNIST.h5", "r") as hf:
        x = np.reshape(np.array(hf.get("x_{}".format(subset)), dtype = np.float32), (-1, 28, 28 ,1))
        y = np.array(hf.get("y_{}".format(subset)))
    if one_hot:
        y = np.eye(10, dtype = np.float32)[np.reshape(y, -1)]
    else:
        y = np.reshape(y, (-1, 1))
    if scaled:
        x = scale(x)
    return x, y

def scale(x):
    x /= 255
    x -= np.mean(x, axis = (1, 2)).reshape(-1, 1, 1, 1)
    x /= np.maximum(np.std(x, axis = (1, 2)).reshape(-1, 1, 1, 1), np.sqrt(x.shape[1] * x.shape[2] - 1))
    return x