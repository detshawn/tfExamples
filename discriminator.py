
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from functools import reduce


class Discriminator:

    def generate(self):
        # PatchGAN Discriminator

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = keras.layers.Input(shape=[None, None, 3], name='target_image')

        x = keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channel*2)

        down_stack = [
            x,
            self.downsample(64, 4, False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
        ]
        x = reduce(lambda y, z: z(y), down_stack)

        zero_pad1 = keras.layers.ZeroPadding2D()(x)  # (bs, 34, 34, 256)
        conv = keras.layers.Conv2D(512, 4, strides=1,
                                   kernel_initializer=initializer,
                                   use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batch_norm1 = keras.layers.BatchNormalization()(conv)

        leaky_relu = keras.layers.LeakyReLU()(batch_norm1)

        zero_pad2 = keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = keras.layers.Conv2D(1, 4, strides=1,
                                   kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return keras.Model(inputs=[inp, tar], outputs=last)

    #  --> Each block in the encoder: Conv -> Batchnorm -> Leaky ReLU
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02) # Theta_ij ~ N(0, 0.02)

        result = keras.Sequential()
        result.add(
            keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(keras.layers.BatchNormalization())

        result.add(keras.layers.LeakyReLU())

        return result
