#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print ("a =2 , b= 3")
    print ("addition: %i" % sess.run(a + b))
    print ("multiplication : %i" % sess.run(a * b))

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print ("add: %d" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
