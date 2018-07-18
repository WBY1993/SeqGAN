# -*- coding: utf-8 -*-
import numpy as np

class Gen_data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        
    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)
                    
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, axis=0)
        self.pointer = 0
        
    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret
        
    def reset_pointer(self):
        self.pointer = 0
        

class Dis_data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        
    def create_batches(self, positive_file, negtive_file):
        positive_examples = []
        negtive_examples = []

        with open(positive_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    positive_examples.append(parse_line)
                
        with open(negtive_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negtive_examples.append(parse_line)
                    
        positive_labels = [[0, 1] for _ in positive_examples]
        negtive_labels = [[1, 0] for _ in negtive_examples]
        self.sentences = np.array(positive_examples + negtive_examples)
        self.labels = np.array(positive_labels + negtive_labels)
        
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]

        self.sentence_batches = np.split(self.sentences, self.num_batch, axis=0)
        self.label_batches = np.split(self.labels, self.num_batch, axis=0)
        self.pointer = 0
        
    def next_batch(self):
        ret = self.sentence_batches[self.pointer], self.label_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret
        
    def reset_pointer(self):
        self.pointer = 0
