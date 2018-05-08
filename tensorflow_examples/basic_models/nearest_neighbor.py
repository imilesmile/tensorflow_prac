#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

# tf graph
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])
#distance 返回的是N个训练样本的和单个测试样本的距离, negative 添加负号
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
pred = tf.argmin(distance, 0)

accuracy = 0.

