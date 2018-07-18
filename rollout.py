# -*- coding: utf-8 -*-
import tensorflow as tf

class rollout():
    def __init__(self):
        self.batch_size = 64
        self.vocab_size = 5000
        self.embedding_size = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.start_token = tf.constant([0] * self.batch_size, dtype=tf.int32)
        self.pre_seq = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="pre_seq")
        self.sample_rollout_step = []
        
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            with tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", shape=[self.hidden_dim, self.vocab_size], initializer=tf.random_normal_initializer)
                
            with tf.variable_scope("lstm"):
                for step in range(1, self.sequence_length):
                    if step % 5 == 0:
                        print("Rollout step: {}".format(step))
                    sample_rollout_left = tf.reshape(self.pre_seq[:, 0:step], shape=[self.batch_size, step])
                    sample_rollout_right = []
                    # left
                    for j in range(step):
                        if j==0:
                            state = lstm1.zero_state(self.batch_size, tf.float32)
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.pre_seq[:, j-1])
                        
                        output, state = lstm1(lstm1_in, state)
                    # right
                    sample_word = self.pre_seq[:, step-1]
                    for j in range(step, self.sequence_length):
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                        output, state = lstm1(lstm1_in, state)# batch * hidden_dim
                        logits = tf.matmul(output, output_W)# batch * vocab_size
                        softmax = tf.nn.softmax(logits)
                        sample_word = tf.reshape(tf.multinomial(tf.log(softmax), 1), shape=[self.batch_size])
                        sample_word = tf.cast(sample_word, dtype=tf.int32)
                        sample_rollout_right.append(sample_word)# seqs-step * batch
                        
                    sample_rollout_right = tf.transpose(sample_rollout_right, perm=[1, 0])# batch * seqs-step
                    sample_roll_out = tf.concat([sample_rollout_left, sample_rollout_right], axis=1)# batch * seqs
                    self.sample_rollout_step.append(sample_roll_out)
                self.sample_rollout_step.append(self.pre_seq)# seqs * batch * seqs
