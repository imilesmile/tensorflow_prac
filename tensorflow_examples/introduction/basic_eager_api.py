#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

print ("setting eager mode...")
tfe.enable_eager_execution()

print ("define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print ("tensor:\n a= %s" % a)

b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)
# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a matmul b = %s" % d)

d = a * b
print("a * b = %s" % d)

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print (a[i][j])
