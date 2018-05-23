#!/bin/python2.7

import os, sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import device_setter

import reader

flags = tf.flags
flags.DEFINE_string("train_path", None, "Where the training data is stored.")
flags.DEFINE_string("infer_path", None, "Where the infer data is stored.")
flags.DEFINE_string("model_path", None, "where the model data is stored.")
flags.DEFINE_string("out_embedding", None, "where the embedding data is stored.")
flags.DEFINE_string("out_softmax", None, "where the softmax weights data is stored.")
flags.DEFINE_string("config", "title", "which config is choosen.")

# cluster
flags.DEFINE_string("ps_hosts", "", "paramater sever hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", "", "job_name: ps or worker")
flags.DEFINE_integer("task_index", 0, "task index")

FLAGS = flags.FLAGS


class TitleConfig(object):
    vocab_size = 10000
    batch_size = 128
    num_steps = 30
    init_scale = 0.05
    lr = 1.0
    lr_decay = 0.5
    lr_nodecay_step = 2
    keep_prob = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 128
    embedding_size = 128
    max_epoch = 5


class ClickConfig(object):
    vocab_size = 1000001
    batch_size = 25
    num_steps = 30
    init_scale = 0.05
    lr = 1.0
    lr_decay = 0.5
    lr_nodecay_step = 3
    keep_prob = 0.8
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 128
    embedding_size = 128
    max_epoch = 5


class Input(object):
    def __init__(self, config, data, is_infer, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        if not is_infer:
            self.epoch_size = (len(data) // batch_size - 1) // num_steps
            self.x, self.y, self.xlen = reader.train_iterator(data, batch_size, num_steps, name)
        else:
            self.epoch_size = len(data) // batch_size
            self.x, self.y, self.xlen = reader.infer_iterator(data, batch_size, num_steps, name)


class SeqModel(object):
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._config = config
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.epoch_size = input_.epoch_size

        vocab_size = config.vocab_size
        embedding_size = config.embedding_size
        hidden_size = config.hidden_size

        self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
        self.inputs = tf.nn.embedding_lookup(self.embedding, input_.x)
        if is_training and config.keep_prob < 1:
            self.inputs = tf.nn.dropout(self.inputs, config.keep_prob)
        self.output, self.state = self._build_rnn_graph(self.inputs, input_.xlen)
        reshape_output = tf.reshape(tf.concat(self.output, 1), [-1, config.hidden_size])

        ## last hidden output
        _, hidden_output = self.state[config.num_layers - 1]
        self._norm_hidden_output = tf.nn.l2_normalize(hidden_output, dim=1)
        ## mean pooling layer
        reshape_xlen = tf.cast(tf.reshape(input_.xlen, [-1, 1]), dtype=tf.float32)
        mean_pooling = tf.divide(tf.reduce_sum(self.output, axis=1), reshape_xlen)
        self._norm_pooling = tf.nn.l2_normalize(mean_pooling, dim=1)

        self.softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
        self.softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        self._norm_softmax_w = tf.nn.l2_normalize(self.softmax_w, dim=0)
        logits = tf.nn.xw_plus_b(reshape_output, self.softmax_w, self.softmax_b)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                input_.y,
                                                tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                                                average_across_timesteps=False,
                                                average_across_batch=True)
        self._cost = tf.reduce_sum(loss)
        self._final_state = self.state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, inputs_len):
        config = self._config
        is_training = self._is_training
        cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                            reuse=not is_training)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([cell] * config.num_layers, state_is_tuple=True)
        self._initial_state = cells.zero_state(config.batch_size, tf.float32)
        output, state = tf.nn.dynamic_rnn(cells, inputs, sequence_length=inputs_len, initial_state=self._initial_state)
        return output, state

    def assign_lr(self, sess, lr_value):
        sess.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def norm_pooling(self):
        return self._norm_pooling

    @property
    def norm_hidden_output(self):
        return self._norm_hidden_output

    @property
    def norm_softmax_w(self):
        return self._norm_softmax_w


def run_epoch(sess, model, eval_op=None, verbose=False):
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)
    fetches = {"cost": model.cost, "final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps
        if step % (model.input.epoch_size // 10) == 10:
            progress = step * 1.0 / model.input.epoch_size
            perplexity = np.exp(costs / iters)
            print("progress: %.3f, perplexity: %.3f" % (progress, perplexity))
            sys.stdout.flush()
    return np.exp(costs / iters)


def infer_pooling(sess, model):
    print "infer pooling"
    poolings = []
    for step in range(model.input.epoch_size):
        state = sess.run(model.initial_state)
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        polling = sess.run(model.norm_pooling, feed_dict)
        for row in polling:
            format_row = ",".join([str(_) for _ in row])
            poolings.append(format_row)
        if step % (model.input.epoch_size // 10) == 10:
            progress = step * 1.0 / model.input.epoch_size
            print("progress: %.3f" % progress)
            sys.stdout.flush()
    return poolings


def infer_hidden_output(sess, model):
    print "infer hidden output"
    outputs = []
    for step in range(model.input.epoch_size):
        state = sess.run(model.initial_state)
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        output = sess.run(model.norm_hidden_output, feed_dict)
        for row in output:
            format_row = ",".join([str(_) for _ in row])
            outputs.append(format_row)
        if step % (model.input.epoch_size // 10) == 10:
            progress = step * 1.0 / model.input.epoch_size
            print("progress: %.3f" % progress)
            sys.stdout.flush()
    return outputs


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    print (worker_hosts, job_name, task_index)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if FLAGS.job_name == "ps":
        server.join()
    else:
        worker_device = "/job:worker/task:%d" % task_index
        ps_strategy = device_setter.GreedyLoadBalancingStrategy(len(ps_hosts), device_setter.byte_size_load_fn)
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                      ps_strategy=ps_strategy,
                                                      cluster=cluster)):

            print ("train_path: ", FLAGS.train_path)
            print ("infer_path: ", FLAGS.infer_path)
            print ("model_path: ", FLAGS.model_path)
            print ("out_embedding: ", FLAGS.out_embedding)
            print ("out_softmax: ", FLAGS.out_softmax)
            print ("config: ", FLAGS.config)

            train_config = TitleConfig() if FLAGS.config == "title" else ClickConfig()
            infer_config = TitleConfig() if FLAGS.config == "title" else ClickConfig()
            test_config = TitleConfig() if FLAGS.config == "title" else ClickConfig()
            test_config.batch_size = 1
            test_config.num_steps = 1
            train_data, valid_data, test_data = reader.read_train(FLAGS.train_path)
            if FLAGS.infer_path:
                infer_keys, infer_data = reader.read_infer(FLAGS.infer_path)

            with tf.Graph().as_default():
                initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
                with tf.name_scope("Train"):
                    train_input = Input(config=train_config, data=train_data, is_infer=False, name="TrainInput")
                    with tf.variable_scope("Model", reuse=None, initializer=initializer):
                        m_train = SeqModel(is_training=True, config=train_config, input_=train_input)
                with tf.name_scope("Valid"):
                    valid_input = Input(config=train_config, data=valid_data, is_infer=False, name="ValidInput")
                    with tf.variable_scope("Model", reuse=True, initializer=initializer):
                        m_valid = SeqModel(is_training=False, config=train_config, input_=valid_input)
                with tf.name_scope("Test"):
                    test_input = Input(config=test_config, data=test_data, is_infer=False, name="TestInput")
                    with tf.variable_scope("Model", reuse=True, initializer=initializer):
                        m_test = SeqModel(is_training=False, config=test_config, input_=test_input)
                if FLAGS.infer_path:
                    with tf.name_scope("Infer"):
                        infer_input = Input(config=infer_config, data=infer_data, is_infer=True, name="InferInput")
                        with tf.variable_scope("Model", reuse=True, initializer=initializer):
                            m_infer = SeqModel(is_training=False, config=infer_config, input_=infer_input)

                with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                                       checkpoint_dir=FLAGS.model_path) as sess:
                    if not FLAGS.infer_path:
                        for i in range(train_config.max_epoch):
                            lr_decay = train_config.lr_decay ** max(i + 1 - train_config.lr_nodecay_step, 0.0)
                            m_train.assign_lr(sess, train_config.lr * lr_decay)
                            print("epoch: %d, learning rate is: %.3f" % (i + 1, sess.run(m_train.lr)))
                            train_perplexity = run_epoch(sess, m_train, eval_op=m_train.train_op)
                            print("Epoch: %d, Train perplexity: %.3f" % (i + 1, train_perplexity))
                            valid_perplexity = run_epoch(sess, m_valid)
                            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                            sys.stdout.flush()
                        test_perplexity = run_epoch(sess, m_test)
                        print("Test Perplexity: %.3f" % test_perplexity)
                        # save softmax weights
                        softmaxs = []
                        softmax = sess.run(m_train.norm_softmax_w)
                        for col in softmax.T:
                            format_col = ",".join([str(_) for _ in col])
                            softmaxs.append(format_col)
                        with open(FLAGS.out_softmax, 'w') as f:
                            for idx in range(len(softmaxs)):
                                ss = str(idx) + "\t" + softmaxs[idx] + "\n"
                                f.write(ss)
                    else:
                        latest_ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
                        sess.restore(sess, os.path.join(FLAGS.model_path, "model.ckpt"))
                        if FLAGS.config == "title":
                            embeddings = infer_pooling(sess, m_infer)
                        else:
                            embeddings = infer_hidden_output(sess, m_infer)
                        with open(FLAGS.out_embedding, "w") as f:
                            for idx in range(len(embeddings)):
                                ss = infer_keys[idx] + "\t" + embeddings[idx] + "\n"
                                f.write(ss)


if __name__ == "__main__":
    tf.app.run()
