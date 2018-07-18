# -*- coding: utf-8 -*-
import tensorflow as tf

class Generator():
    def __init__(self):
        self.batch_size = 64
        self.vocab_size = 5000
        self.embedding_size = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.start_token = tf.constant([0] * self.batch_size, dtype=tf.int32)
        
    def build_input(self, name):
        if name == "pretrain":
            self.input_seqs_pre = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_seqs_pre")
            self.input_seqs_mask = tf.placeholder(tf.float32, shape=[None, self.sequence_length], name="input_seqs_mask")
        elif name == "adversarial":
            self.input_seqs_adv = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_seqs_adv")
            self.rewards = tf.placeholder(tf.float32, shape=[None, self.sequence_length], name="rewards")
            
    def build_pretrain_network(self):
        self.build_input(name="pretrain")
        self.pretrained_loss = 0
        with tf.variable_scope("teller"):
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", shape=[self.hidden_dim, self.vocab_size], initializer=tf.random_normal_initializer)
                
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    if j==0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_pre[:, j-1])
                        
                    output, state = lstm1(lstm1_in, state)# batch * hidden_dim
                    logits = tf.matmul(output, output_W)# batch * vocab_size
                    
                    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs_pre[:, j], logits=logits)
                    pretrained_loss_t = tf.reduce_sum(tf.multiply(pretrained_loss_t, self.input_seqs_mask[:, j]))
                    self.pretrained_loss += pretrained_loss_t
                    
                self.pretrained_loss /= self.sequence_length
                self.pretrained_loss_sum = tf.summary.scalar("pretrained_loss", self.pretrained_loss)
                
    def build_adversarial_network(self):
        self.build_input(name="adversarial")
        self.softmax_list = []

        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", shape=[self.hidden_dim, self.vocab_size], initializer=tf.random_normal_initializer)
            
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    if j==0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_adv[:, j-1])
                        
                    output, state = lstm1(lstm1_in, state)# batch * hidden_dim
                    logits = tf.matmul(output, output_W)# batch * vocab_size
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax)# seqs * batch * vocab_size
                    
            self.softmax_list_reshape = tf.transpose(self.softmax_list, perm=[1, 0, 2])# batch * seqs * vocab_size
            cross_entropy = -tf.reduce_sum(tf.one_hot(tf.reshape(self.input_seqs_adv, [-1]), self.vocab_size) * tf.log(tf.reshape(self.softmax_list_reshape, [-1, self.vocab_size])), axis=1)
            self.adversarial_loss = tf.reduce_sum(cross_entropy * tf.reshape(self.rewards, [-1]))
            
    def build_sample_network(self):
        self.sample_word_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", shape=[self.hidden_dim, self.vocab_size], initializer=tf.random_normal_initializer)
                
            with tf.variable_scope("lstm"):
                sample_word = self.start_token
                state = lstm1.zero_state(self.batch_size, tf.float32)
                for j in range(self.sequence_length):
                    lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                    output, state = lstm1(lstm1_in, state)# batch * hidden_dim
                    logits = tf.matmul(output, output_W)# batch * vocab_size
                    softmax = tf.nn.softmax(logits)
                    sample_word = tf.reshape(tf.multinomial(tf.log(softmax), 1), shape=[self.batch_size])
                    self.sample_word_list.append(sample_word)# seqs * batch
                    
            self.sample_word_list_reshape = tf.transpose(self.sample_word_list, perm=[1, 0])# batch * seqs
            
    def build(self):
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()
        
    def generate(self, sess):
        return sess.run(self.sample_word_list_reshape)
