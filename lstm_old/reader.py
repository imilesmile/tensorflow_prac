#!/bin/python2.7

import collections
import os
import sys

import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().strip().replace("\n", ",0,").split(",")

def read_train(data_path):
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")
    train_data = _read_words(train_path)
    train_data = [int(_) for _ in train_data]
    valid_data = _read_words(valid_path)
    valid_data = [int(_) for _ in valid_data]
    test_data = _read_words(test_path)
    test_data = [int(_) for _ in test_data]

    return train_data, valid_data, test_data

def read_infer(data_path):
    infer_path = os.path.join(data_path, "infer")
    allkeys = []
    allvalues = []
    with tf.gfile.GFile(infer_path, "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            value_arr = value.strip().split(",")
            value_arr = [int(_) for _ in value_arr]
            allkeys.append(key)
            allvalues.append(value_arr)

    return allkeys, allvalues

def train_iterator(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "train_iterator", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        x_len = tf.fill([batch_size], num_steps)
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y, x_len

def infer_iterator(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "infer_iterator", [raw_data, batch_size, num_steps]):
        data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        epoch_size = tf.size(data) // batch_size
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [i*batch_size, 0], [(i+1)*batch_size, num_steps])
        x.set_shape([batch_size, num_steps])
        x_len = tf.count_nonzero(x, axis=1)
        return x, x, x_len

