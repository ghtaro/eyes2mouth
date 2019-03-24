from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import csv
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from ops import *
from utils import *

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_c_dim + self.output_c_dim], \
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim] # upper-half:real, lower-half:blank(in black)
        self.real_AB = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim] # entirely real image

        # generate a full face image given real_A
        self.fake_AB = self.generator(self.real_A)

        #self.real_AB = tf.concat([self.real_A, self.real_B], 2)
        #self.fake_AB = tf.concat([self.real_A, self.fake_B], 2)
        # D(return of the sigmoid: probability), D_logits(before sigmoid), D_h3(matrix passed to linear+sigmoid)
        self.D, self.D_logits, self.D_h3 = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_, self.D_h3_ = self.discriminator(self.fake_AB, reuse=True)

        # sampling by using valiation data 
        self.fake_AB_sample = self.sampler(self.real_A)

        # logging - No idea how to check these values afterwards...
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_AB_sum = tf.summary.image("fake_AB", self.fake_AB)
        
        # Cross entropy loss function
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_AB - self.fake_AB)) # added a term proportional to |real_AB - fake_AB| for whatever reasons

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        # total d_loss
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        # collect all model weights (beginning with d_ and g_) defined in this TF graph
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.saver = tf.train.Saver()

    def load_random_samples(self):
        data = np.random.choice(glob('/root/userspace/eye2mouth/datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file, fine_size=self.image_size) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def proess_weights(self, d_h3):
        ws = [v for v in tf.trainable_variables() if v.name == 'discriminator/d_h3_lin/Matrix:0'][0]
        ws = ws.eval() # tf.Tensor -> np array
        ws = ws.flatten() # np array (n,1) -> (n)
        ws = np.reshape(ws, (16,16,512)) # np array reshape
        b = [v for v in tf.trainable_variables() if v.name == 'discriminator/d_h3_lin/bias:0'][0]
        b = b.eval()
        sample_weights = np.sum(ws * d_h3 + b, axis=3) # sum [batch_size,16,16,512] along the axis=3
        sample_weights = sample_weights[:,:,:,np.newaxis] # add new axis (with size 1) for color
        sample_weights = np.insert(sample_weights, 1, 0, axis=3)
        sample_weights = np.insert(sample_weights, 2, 0, axis=3)
        sample_weights = rot90_batch(sample_weights, 3) # [batch_size, 16, 16, 512]
        return sample_weights
    
    def save_real_fake_images_weights(self, real_images, fake_images, real_weights, fake_weights, sample_dir, epoch, idx, D_lgt, D_lgt_):
        all_resize = []
        for sr, sf, r, f in zip(real_images, fake_images, real_weights, fake_weights):
            r0 = r[:,:,0]  # only first-dim is filled with non-zero value
            r_max = (r0.max(axis=0)).max(axis=0)
            r_min = (r0.min(axis=0)).min(axis=0)
            f0 = f[:,:,0]  # only first-dim is filled with non-zero value
            f_max = (f0.max(axis=0)).max(axis=0)
            f_min = (f0.min(axis=0)).min(axis=0)
            s_max = np.max((r_max, f_max, np.abs(r_min), np.abs(f_min)))
            r_pos = np.where(r0 > 0, r0, 0) / s_max * 255.0 # positive part and normalised [0,255]
            r_neg = np.where(r0 < 0, r0, 0) / (-s_max) * 255.0 # negative part and normalised [0,255]
            f_pos = np.where(f0 > 0, f0, 0) / s_max * 255.0 # positive part and normalised [0,255]
            f_neg = np.where(f0 < 0, f0, 0) / (-s_max) * 255.0 # negative part and normalised [0,255]

            r[:,:,0] = r_neg # Red for negative
            r[:,:,1] = 0 # Green if s0 is 0
            r[:,:,2] = r_pos # Blue for positive
            r[0,:,] = 255.0 # border to white
            r = r.astype(np.uint8) # to adapt to nparray to image

            f[:,:,0] = f_neg # Red for negative
            f[:,:,1] = 0 # Green if s0 is 0
            f[:,:,2] = f_pos # Blue for positive
            f[0,:,] = f[:,0,] = 255.0 # border to white
            f = f.astype(np.uint8) # to adapt to nparray to image
            
            w = sr.shape[0]
            wu = int(w/2)
            srf = np.concatenate((sr[:wu, :], sf[wu:, :]), axis=0)
            s1 = np.concatenate([sr, sf, srf], axis=1)
            s1 = scipy.misc.imresize(s1, (self.image_size, self.image_size*3))
            s1_image = Image.fromarray(s1, 'RGB')
            s2 = np.concatenate([r, f], axis=1)
            s2_image = Image.fromarray(s2, 'RGB')
            s2_image = s2_image.resize((self.image_size*2, self.image_size))
            s_image = get_concat_h(s1_image, s2_image)
            all_resize.append(np.array(s_image))
            
        all_resize = np.array(all_resize)
        save_images(all_resize, [self.batch_size, 1], '/root/userspace/eye2mouth/{}/{}_{:02d}_{:04d}.png'.format(sample_dir, 'faces_and_weights', epoch, idx))
        with open('/root/userspace/eye2mouth/{}/log_{:02d}_{:04d}.csv'.format(sample_dir, epoch, idx), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(np.concatenate([np.array(D_lgt), np.array(D_lgt_)], axis=1))

    def sample_model(self, sample_dir, epoch, idx, save_weights):
        # Get valiation data
        sample_images = self.load_random_samples()
        # Run the TF graph and get the variables we want
        samples_fake, d_loss, g_loss, d_h3, d_h3_, D_lgt, D_lgt_ = self.sess.run([self.fake_AB_sample, self.d_loss, self.g_loss, self.D_h3, self.D_h3_, self.D_logits, self.D_logits_],\
                                                                            feed_dict={self.real_data: sample_images})
        if save_weights:
            sample_real = sample_images[:,:,:,3:]
            fake_weights = self.proess_weights(d_h3_)
            real_weights = self.proess_weights(d_h3)
            self.save_real_fake_images_weights(sample_real, samples_fake, real_weights, fake_weights, sample_dir, epoch, idx, D_lgt, D_lgt_)

        else:
            samples_fake = imresize_batch(samples_fake, (self.image_size, self.image_size))
            save_images(samples_fake, [self.batch_size, 1], '/root/userspace/eye2mouth/{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lrd, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lrg, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.fake_AB_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("/root/userspace/eye2mouth/logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        log_loss = []
        for epoch in xrange(args.epoch):
            data = glob('/root/userspace/eye2mouth/datasets/{}/train/*.jpg'.format(self.dataset_name))
            #np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, fine_size=self.image_size) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Try three time g_optim (my change. I want to reduce the effect of L1 norm.)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
                log_loss.append([epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG])

                if np.mod(counter, args.sample_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, args.sample_weights)

                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)
        
        with open('/root/userspace/eye2mouth/log.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(np.array(log_loss))

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4, h3

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e6], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e5], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e4], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e3], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s4, s4, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e2], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s2, s2, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e1], 3)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s, s, self.output_c_dim], name='g_d7', with_w=True)
            # d8 is (128 x 128 x output_c_dim)

            return tf.nn.tanh(self.d7)

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e6], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e5], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e4], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e3], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s4, s4, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e2], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s2, s2, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e1], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s, s, self.output_c_dim], name='g_d7', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d7)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s" % (self.dataset_name, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        from scipy.misc import imresize, imread, imsave
        from utils import rot90_batch, imresize_batch
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('/root/userspace/eye2mouth/datasets/{}/val/*.jpg'.format(self.dataset_name))

        # sort testing input
        # n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        n = range(len(sample_files))
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True, fine_size=self.image_size) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            sample_image_a = sample_image[:, :, :, :3]
            samples_concat = np.concatenate((sample_image_a, samples), axis=2)
            samples_concat = imresize_batch(samples_concat, (self.image_size, self.image_size))
            samples_concat = rot90_batch(samples_concat, 3)
            print('saving them in /root/userspace/eye2mouth/{}/'.format(args.test_dir))
            save_images(samples_concat, [self.batch_size, 1],
                        '/root/userspace/eye2mouth/{}/test_{:04d}.png'.format(args.test_dir, idx))
