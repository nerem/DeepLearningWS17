import numpy as np
import tensorflow as tf
import preparation

def generate_model(n, lam = 0, step_size = 0.01):

	# Define model and initialize variables.
	W = tf.Variable(-np.ones((n, 1)), dtype=tf.float32)
	b = tf.Variable([1.], dtype=tf.float32)
	x = tf.placeholder(tf.float32, shape=[None, n + 1])
	linear_model = x[:,0] * (x[:,1:] @ W + b)
	y = tf.placeholder(tf.float32)

	# Define loss function and choose optimizer.
	l2_loss = (lam/2) * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

	loss = tf.reduce_mean(tf.square((tf.clip_by_value(linear_model, 0, 255) - y) / 255.))
	loss_eval = loss
	loss = tf.add(loss, l2_loss)
	optimizer = tf.train.GradientDescentOptimizer(step_size)
	train = optimizer.minimize(loss)

	# Initialize tensorflow session.
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	return sess, W, b, x, y, loss, loss_eval, optimizer, train, init

# perform stochastic gradient descent
def sgd(sess, x, y, train, loss, x_train, y_train, x_val, y_val, minibatch_size = 1000):

	# create minibatch for sgd
	index = batch(minibatch_size, x_train.shape[0])
	for i in range(int(x_train.shape[0]/minibatch_size)-1):
		# train on batches until all data is used
		sess.run(train, {x: x_train[index==i+1], y: y_train[index==i+1]})

	loss_val = calculate_loss(sess, x, y, loss, x_val, y_val, minibatch_size)
	print("Validation loss: ", loss_val)

	return loss_val


# splits data in total/size batches
def batch(size, total):
    index = np.zeros(total)
    for i in range(int(total/size)):
        index[i*size:(i+1)*size] = i+1
    index = np.random.permutation(index)

    return index

# perform forward pass for each mini batch in valuation data
def calculate_loss(sess, x, y, loss, x_eval, y_eval, batch_size = 1000):
	
	losses = []
	for i in range(int(x_eval.shape[0] / batch_size)):
		losses.append(sess.run(loss, {x: x_eval[i*batch_size:(i+1)*batch_size], y: y_eval[i*batch_size:(i+1)*batch_size]}))

	return np.mean(losses)


# perform k-fold cross validation
def cross_validation(sess, init, x, y, train, loss, loss_eval, X, Y, minibatch_size = 1000, epochs = 2, k = 3):

	index = batch(int(X.shape[0] / k), X.shape[0])
	cross_loss = []
	for i in range(k):
		sess.run(init)
		print("")
		print("Cross validation: ", (i+1), "/", k)
		for j in range(epochs):
			print("Epoch: ", (j+1), "/", epochs)
			sgd(sess, x, y, train, loss, X[index!=i+1], Y[index!=i+1], X[index==i+1], Y[index==i+1], minibatch_size)
		cross_loss.append(calculate_loss(sess, x, y, loss_eval, X[index==i+1],  Y[index==i+1], minibatch_size))

	return np.mean(cross_loss)


# training on complete training data
def complete(n = 2, lam = 0, epochs = 2, minibatch_size = 1000, ratio = 0.8, step_size = 0.01):

	# Prepare data.
	imgs = ["cat_01.jpg", "cat_02.jpg", "cat_03.jpg"]
	imgs_vignetted = ["cat_01_vignetted.jpg", "cat_02_vignetted.jpg", "cat_03_vignetted.jpg"]
	X, Y = preparation.multi_prep(imgs, imgs_vignetted, n)

	# Split data int training and test sets.
	X_Train, Y_Train, X_Test, Y_Test = preparation.split(X, Y, ratio = ratio)

	# Initialize the model.
	sess, W, b, x, y, loss, loss_eval, optimizer, train, init = generate_model(n, lam, step_size = step_size)

	# Split the data into training and validation sets.
	x_train, y_train, x_val, y_val = preparation.split(X_Train, Y_Train, ratio = ratio)

	# Train
	for i in range(epochs):
		print("")
		print("Epoch: ", (i+1), "/", epochs)
		sgd(sess, x, y, train, loss, x_train, y_train, x_val, y_val, minibatch_size)
		calculate_loss(sess, x, y, loss_eval, x_val, y_val, batch_size = minibatch_size)

	return sess.run([W, b])

# training cross validation: split training data again in training and validation data
def complete_cv(n = 2, lam = 0, epochs = 2, k = 1, minibatch_size = 1000, ratio = 0.8, step_size = 0.01):

	# Prepare data.
	imgs = ["cat_01.jpg", "cat_02.jpg", "cat_03.jpg"]
	imgs_vignetted = ["cat_01_vignetted.jpg", "cat_02_vignetted.jpg", "cat_03_vignetted.jpg"]
	X, Y = preparation.multi_prep(imgs, imgs_vignetted, n)

	# Split data int training and test sets.
	X_Train, Y_Train, X_Test, Y_Test = preparation.split(X, Y, ratio = ratio)

	# Initialize the model.
	sess, W, b, x, y, loss, loss_eval, optimizer, train, init = generate_model(n, lam, step_size = step_size)

	# Split the data into training and validation sets.
	x_train, y_train, x_val, y_val = preparation.split(X_Train, Y_Train, ratio = ratio)

	# Train
	cross_loss = cross_validation(sess, init, x, y, train, loss, loss_eval, X_Train, Y_Train, minibatch_size = minibatch_size, epochs = epochs, k = k)

	#print("")
	print("Result of cross validation for n =", n)
	print("Loss:", cross_loss)
	#print("")

	return cross_loss