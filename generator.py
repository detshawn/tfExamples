
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras


class Generator:

    def generate(self):
        OUTPUT_CHANNEL = 3

        initializer = tf.random_normal_initializer(0., 0.02)


        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            self.upsample(512, 4), # (bs, 16, 16, 1024)
            self.upsample(256, 4), # (bs, 32, 32, 512)
            self.upsample(128, 4), # (bs, 64, 64, 256)
            self.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        inputs = keras.layers.Input(shape=[None, None, 3])
        concat = keras.layers.Concatenate()
        last = keras.layers.Conv2DTranspose(OUTPUT_CHANNEL, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (bs, 256, 256, 3)

        #  --> Skip connections btw the encoder and the decoder
        x = inputs

        # Downsampling via the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])  # get rid of the last block and reverse the list

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return keras.Model(inputs=inputs, outputs=x)

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

    #  --> Each block in the decoder: Transposed Conv -> Bacthnorm -> Dropout (applied to the first 3 blocks) -> ReLU
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = keras.Sequential()
        result.add(
            keras.layers.Conv2DTranspose(filters, size, strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False))

        if apply_dropout:
            result.add(keras.layers.Dropout(0.5))

        result.add(keras.layers.ReLU())

        return result

    def test_code(self, inp):
        down_model = self.downsample(3, 4)
        down_result = down_model(tf.expand_dims(inp, 0))
        print (down_result.shape)

        up_model = self.upsample(3, 4)
        up_result = up_model(down_result)
        print (up_result.shape)
