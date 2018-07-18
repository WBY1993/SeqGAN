# -*- coding: utf-8 -*-
import tensorflow as tf


def linear(x, output_size, scope=None):
    shape = x.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear expects 2D arguments: %s" % str(shape))
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", shape=[output_size, shape[1]], dtype=x.dtype)
        bias_term = tf.get_variable("Bias", shape=[output_size], dtype=x.dtype)
    return tf.matmul(x, tf.transpose(matrix)) + bias_term


def highway(x, size, num_layers=1, bias=-2.0, scope="Highway"):
    with tf.variable_scope("scope"):
        for idx in range(num_layers):
            g = tf.nn.relu(linear(x, size, scope="highway_lin_%d" % idx))
            t = tf.nn.sigmoid(linear(x, size, scope="highway_gate_%d" % idx) + bias)
            output = t*g + (1.0-t)*x
            x = output
    return output
    

class Discriminator():
    def __init__(self):
        self.sequence_length =  20
        self.num_class = 2
        self.vocab_size = 5000
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]# convolutional kernel size of discriminator
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]# number of filters of each conv kernel
        self.learning_rate = 1e-4
        self.embedding_size = 64
        self.l2_reg_lambda = 0.2
        self.input_data = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_data")
        self.input_label = tf.placeholder(tf.int32, shape=[None, self.num_class], name="input_label")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.l2_loss = tf.constant(0.0)
        
    def build(self):
        with tf.variable_scope("discriminator"):
            with tf.name_scope("embedding"):
                self.W = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_data)# batch * seq * emb_size
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)# batch * seq * emb_size * 1
                
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")# batch * seq - filter_size + 1 * 1 * num_filter
                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length-filter_size+1, 1, 1], strides=[1,1,1,1], padding="VALID", name="pool")# batch * 1 * 1 * num_filter
                    pooled_outputs.append(pooled)
            
            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])# batch * sum_num_fiters
            
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, num_filters_total, 1, 0)
                
            with tf.name_scope("drop_out"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.keep_prob)
                
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_class], mean=0.0, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")# batch * num_classes
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.ypred_for_auc, axis=1, name="predictions")
            
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.scores)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*self.l2_loss

        self.params = [param for param in tf.trainable_variables() if "discriminator" in param.name]
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
