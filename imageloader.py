
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os

import matplotlib.pyplot as plt


class ImageLoader:

    def __init__(self, url, folder_name, img_height=256, img_width=256):
        self._URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

        path_to_zip = tf.keras.utils.get_file(self._URL.split('/')[-1],
                                              origin=self._URL,
                                              extract=True)
        self.PATH = os.path.join(os.path.dirname(path_to_zip), folder_name+('/' if folder_name[-1] != '/' else ''))
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256

    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)  # decode the jpeg to ?

        w = tf.shape(image)[1]  # the shape of the given tensor, i.e., the dim info.

        w = w // 2  # floor division, i.e., it returns only the quotient of division
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image


    @tf.function()
    def random_jitter(self, input_image, real_image):
        # resizing to 286 x 286 x 3
        input_image, real_image = self.resize(input_image, real_image, 286, 286)

        # randomly cropping to img_height x img_width x 3
        input_image, real_image = self.random_crop(input_image, real_image)

        # random mirroring
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image,
                                              self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def test_code(self):
        # loading a jpeg image
        inp, re = self.load(self.PATH + 'train/100.jpg')
        plt.figure()
        plt.imshow(inp / 255.0)
        plt.figure()
        plt.imshow(re / 255.0)

        # random jittering test
        plt.figure(figsize=(6, 6))
        for i in range(4):
            rj_inp, rj_re = self.random_jitter(inp, re)
            plt.subplot(2, 2, i+1)
            plt.imshow(rj_inp/255.0)
            plt.axis('off')
        plt.show()

        inp2, re2 = self.load_image_train(self.PATH + 'train/100.jpg')
        print('inp2, re2 shapes: ', inp2.shape, re2.shape)

        ret = input()
