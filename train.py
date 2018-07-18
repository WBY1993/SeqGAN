# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import os
import dataloader
import discriminator
import generator
import rollout
import utils
import target_lstm

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.70
os.environ["CUDA_VISIBLE_DIVICES"] = "0"

def main(unused_argv):
    gen_data_loader = dataloader.Gen_data_loader(64)
    dis_data_loader = dataloader.Dis_data_loader(64)
    likelihood_data_loader = dataloader.Gen_data_loader(64)
    
    gen_model = generator.Generator()
    gen_model.build()
    dis_model = discriminator.Discriminator()
    dis_model.build()
    rollout_model = rollout.rollout()
    
    target_params = pickle.load(open("save/target_params.pkl", "rb"), encoding="iso-8859-1")
    target_lstm_model = target_lstm.TARGET_LSTM(params=target_params)
    
    # Build optimizer op for pretraining
    pretrained_optimizer = tf.train.AdamOptimizer(1e-2)
    var_pretrained = [v for v in tf.trainable_variables() if "teller" in v.name]
    gradients1, variables1 = zip(*pretrained_optimizer.compute_gradients(gen_model.pretrained_loss, var_list=var_pretrained))
    gradients1, _ = tf.clip_by_global_norm(gradients1, 5.0)
    gen_pre_update = pretrained_optimizer.apply_gradients(zip(gradients1, variables1))
    
    # Build optimizer op for adversarial training
    advtrained_optimizer = tf.train.AdamOptimizer(1e-2)
    gradients2, variables2 = zip(*advtrained_optimizer.compute_gradients(gen_model.adversarial_loss, var_list=var_pretrained))
    gradients2, _ = tf.clip_by_global_norm(gradients2, 5.0)
    gen_adv_update = advtrained_optimizer.apply_gradients(zip(gradients2, variables2))
    
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    log = open("save/log.txt", "w", encoding="utf-8")
    
    print("Start pre-training generator.....")
    utils.generate_samples(sess, target_lstm_model, 64, 1000, "save/real_data.txt")
    gen_data_loader.create_batches("save/real_data.txt")
    log.write("Start pre-training generator.....\n")
    for epoch in range(20):
        gen_data_loader.reset_pointer()
        for it in range(gen_data_loader.num_batch):
            batch = gen_data_loader.next_batch()
            _, g_loss = sess.run([gen_pre_update, gen_model.pretrained_loss],
                                 feed_dict={gen_model.input_seqs_pre: batch,
                                            gen_model.input_seqs_mask:np.ones_like(batch)})
            
        if epoch % 5 == 0:
            utils.generate_samples(sess, gen_model, 64, 1000, "save/eval_data.txt")
            likelihood_data_loader.create_batches("save/eval_data.txt")
            test_loss = utils.target_loss(sess, target_lstm_model, likelihood_data_loader)
            print("epoch:", epoch, " test_loss:", test_loss)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
            
    print("Start pre-training discriminator.....")
    for t in range(2):
        print("Times: " + str(t))
        utils.generate_samples(sess, gen_model, 64, 1000, "save/generate_sample.txt")
        dis_data_loader.create_batches("save/real_data.txt", "save/generate_sample.txt")
        for epoch in range(20):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                sess.run(dis_model.train_op,
                         feed_dict={dis_model.input_data: x_batch,
                                    dis_model.input_label: y_batch,
                                    dis_model.keep_prob: 0.5})
                
    print("Start adversarial training.....")
    for total_batch in range(20):
        for gen_step in range(1):
            samples = sess.run(gen_model.sample_word_list_reshape)
            reward_rollout = []
            for rollout_num in range(16):
                rollout_list = sess.run(rollout_model.sample_rollout_step, feed_dict={rollout_model.pre_seq: samples})
                rollout_list_stack = np.vstack(rollout_list)
                reward_rollout_seq = sess.run(dis_model.ypred_for_auc,
                                              feed_dict={dis_model.input_data: rollout_list_stack,
                                                         dis_model.keep_prob: 1.0})
                reward_all_seq = reward_rollout_seq[:, 1]
                reward_tmp = []
                for r in range(64):
                    reward_tmp.append(reward_all_seq[range(r, 64*20, 64)])
                reward_rollout.append(np.array(reward_tmp))
            
            rewards = np.mean(reward_rollout, axis=0)
            _, gen_loss = sess.run([gen_adv_update, gen_model.adversarial_loss],
                                   feed_dict={gen_model.input_seqs_adv: samples,
                                              gen_model.rewards: rewards})
            
        if total_batch % 5 == 0 or total_batch == 199:
            utils.generate_samples(sess, gen_model, 64, 1000, "save/eval_data.txt")
            likelihood_data_loader.create_batches("save/eval_data.txt")
            test_loss = utils.target_loss(sess, target_lstm_model, likelihood_data_loader)
            print ('total_batch:', total_batch, ' test_loss:', test_loss)
            buffer = 'total_batch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
            
        for dis_step in range(5):
            utils.generate_samples(sess, gen_model, 64, 1000, "save/generate_sample.txt")
            dis_data_loader.create_batches("save/real_data.txt", "save/generate_sample.txt")
            for epoch in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    sess.run(dis_model.train_op,
                             feed_dict={dis_model.input_data: x_batch,
                                        dis_model.input_label: y_batch,
                                        dis_model.keep_prob: 0.5})
                    
    log.close()
    sess.close()
    
    
    
if __name__ == "__main__":
    tf.app.run()
