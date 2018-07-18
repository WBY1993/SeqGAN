# -*- coding: utf-8 -*-
import numpy as np


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
        
    with open(output_file, "w", encoding="utf-8") as f:
        for line in generated_samples:
            buffer =" ".join([str(x) for x in line]) + "\n"
            f.write(buffer)
            

def target_loss(sess, target_lstm, data_loader):
    nll = []
    data_loader.reset_pointer()
    
    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)
        
    return np.mean(nll)
    