import tensorflow as tf
import math

class WeightedItem2VecBase(object):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 cbow_window,
                 num_sampled,
                 init_embeddings):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_sampled = num_sampled
        self.cbow_window = cbow_window
        self.init_embeddings = init_embeddings
        pass

    def init_placeholder(self):
        self.train_inputs = tf.placeholder(tf.int64, shape=[None, 2 * self.cbow_window])
        self.train_labels  = tf.placeholder(tf.int64, shape=[None])
        self.train_weight = tf.placeholder(tf.float32, shape=[None, 2 * self.cbow_window])
        pass

    def init_variables(self):
        self.embeddings = tf.get_variable
        self.embeddings = tf.Variable(self.init_embeddings)
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.emb_dim],
                                                           stddev=1.0 / math.sqrt(self.emb_dim)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    def init_output(self):
        print "weight shape", self.train_weight.get_shape()
        print "input shape", self.train_inputs.get_shape()
        self.embeded = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        print "embeded shape", self.embeded.get_shape()
        weight = tf.expand_dims(self.train_weight, axis=-1)
        print "weight shape", weight.get_shape()
        # weight = tf.tile(weight, [1, 1, self.emb_dim])
        print "weight shape", weight.get_shape()
        self.sum_weight = tf.reduce_sum(self.train_weight, axis=1)
        self.sum_weight = tf.expand_dims(self.sum_weight, -1)
        # self.sum_weight = tf.tile(tf.expand_dims(self.sum_weight, -1), [1, self.emb_dim])
        self.avg_emb = tf.reduce_sum(self.embeded * weight, axis=1)
        print "avg_emb shape", self.avg_emb.get_shape()
        print "sum_weight", self.sum_weight.get_shape()
        print "weight shape", weight.get_shape()
        self.avg_emb = self.avg_emb / self.sum_weight
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=tf.expand_dims(self.train_labels, -1),
                           inputs=self.avg_emb,
                           num_sampled=self.num_sampled,
                           num_classes=self.vocab_size))

    def init_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def build_graph(self):
        self.init_placeholder()
        self.init_variables()
        self.init_output()
        self.init_optimizer()
