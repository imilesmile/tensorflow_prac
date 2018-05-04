#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

# Training Data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
n_samples = len(train_X)

# param
learning_rate = 0.01
display_step = 100
num_steps = 10000

# weight and bias
W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


# linear regression(Wx+b)
def linear_regression(inputs):
    return inputs * W + b


# mean square error
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)


# sgd optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# initial cost
print("Initial cost= {:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
      "W=", W.numpy(), "b=", b.numpy())

# training
for step in range(num_steps):
    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))
    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
              "W=", W.numpy(), "b=", b.numpy())

#graph display
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, np.array(W * train_X + b), label='Fitted line')
plt.legend()
plt.show()
