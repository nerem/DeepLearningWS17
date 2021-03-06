import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    
    x = features["x"]
    
    conv1 = tf.layers.conv2d(inputs = x, filters = 32, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
    conv3 = tf.layers.conv2d(inputs = pool1, filters = 32, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs = conv3, filters = 16, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2, 2], strides = 2)
    flat1 = tf.layers.flatten(inputs = pool2)
    dense1 = tf.layers.dense(inputs = flat1, units = 1024)
    dropout1 = tf.layers.dropout(inputs = dense1, rate = 0.4, training = (mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs = dropout1, units = 10)
    predictions = tf.nn.softmax(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = {"y": predictions})
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = logits, reduction = tf.losses.Reduction.MEAN)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    kernel1 = tf.get_default_graph().get_tensor_by_name("conv2d/kernel:0")
    tf.summary.image("conv1_kernel", image_grid(kernel1, 6), max_outputs = 1)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = eclr(params["learning_rate"], global_step, 60000, int(x.shape[0]), cycle_epochs = 4, steps_per_cycle = 4)
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1))}
    
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


def clr(initial_rate, global_step, samples_per_epoch, batch_size, cycle_epochs = 10, steps_per_cycle = 5, decay_rate = 0.5):
    cycle_length = int(samples_per_epoch / batch_size * cycle_epochs)
    global_step = tf.floormod(global_step, cycle_length)
    return tf.train.exponential_decay(initial_rate, global_step, int(cycle_length / steps_per_cycle), decay_rate, staircase = True)

def eclr(initial_rate, global_step, samples_per_epoch, batch_size, cycle_epochs = 10, steps_per_cycle = 5, decay_rate = 0.5):
    cycle_length = int(samples_per_epoch / batch_size * cycle_epochs)
    decaying_rate = tf.train.exponential_decay(initial_rate, global_step, cycle_length, decay_rate, staircase = True)
    global_step_mod = tf.floormod(global_step, cycle_length)
    return tf.train.exponential_decay(decaying_rate, global_step_mod, int(cycle_length / steps_per_cycle), decay_rate,
                                      staircase = True)

def image_grid(kernel, r):
    p = int(r * np.ceil(int(kernel.shape[3]) / r)) - int(kernel.shape[3])
    kernel -= tf.reduce_min(kernel)
    alpha = tf.pad(tf.constant(1, dtype = tf.float32, shape = kernel.shape), [[0, 1], [0, 1], [0, 0], [0, p]])
    kernel = tf.pad(kernel, [[0, 1], [0, 1], [0, 0], [0, p]])
    s = kernel.shape
    kernel_image = tf.pad(tf.reshape(tf.transpose(tf.reshape(kernel, [s[0], s[1], r, -1]), [2, 0, 3, 1]), [1, s[0] * r, -1, 1]),
                          [[0, 0], [1, 0], [1, 0], [0, 0]])
    alpha_image = tf.pad(tf.reshape(tf.transpose(tf.reshape(alpha, [s[0], s[1], r, -1]), [2, 0, 3, 1]), [1, s[0] * r, -1, 1]),
                         [[0, 0], [1, 0], [1, 0], [0, 0]])
    image = tf.constant(0, dtype = tf.float32, shape = [kernel_image.shape[0], kernel_image.shape[1], kernel_image.shape[2], 4])
    return tf.concat([kernel_image, kernel_image, kernel_image, alpha_image], axis = 3)
    














