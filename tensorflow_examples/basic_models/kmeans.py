#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import data
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("./data", one_hot=True)
full_data_x = minst.train.images

# param
num_steps = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 784  # 28x28

# input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

# kmeans pram
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

# build kmeans graph
training_graph = kmeans.training_graph()

# Build KMeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6:  # Tensorflow 1.4+
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

init_vars = tf.initialize_all_variables()

sess = tf.Session()

# run initialize
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    print (idx)
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# assign a label to each centroid
# count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += minst.train.labels[i]

# assign the most freq label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# evaluation ops
cluster_labels = tf.nn.embedding_lookup(labels_map, cluster_idx)
# compute accuracy
correct_predicion = tf.equal(cluster_labels, tf.cast(tf.argmax(y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_predicion, tf.int32))

# test model
test_x, test_y = minst.test.images, minst.test.labels
print ("test accuracy: ", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))
