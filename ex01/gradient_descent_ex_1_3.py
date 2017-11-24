# Verify gradient descent from exercise 1.3 (a)

import tensorflow as tf

x = tf.Variable(2, name='x', dtype=tf.float32)
y = tf.Variable(1, name='y', dtype=tf.float32)
z = tf.Variable(0, name='z', dtype = tf.float32)

E_1 = 2*x**2 + 4*y
E_sum = y + z
E_2 = tf.maximum(tf.zeros(1), E_sum)
E = E_1 + E_2

# gradient descent with learning rate tau = 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(E)

# Initialize tensorflow session.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("Initial theta = ", sess.run(x), sess.run(y), sess.run(z))

# perform two gradient descent steps
sess.run(train)
print("Step 1, theta = ", sess.run(x), sess.run(y), sess.run(z))
sess.run(train)
print("Step 2, theta = ", sess.run(x), sess.run(y), sess.run(z))
