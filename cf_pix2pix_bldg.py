# Pix2Pix
from __future__ import absolute_import, division, print_function, unicode_literals

# 0. Import Libs
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os

from imageloader import ImageLoader
from generator import Generator
from discriminator import Discriminator
import time
from IPython.display import clear_output


# 1. Import Data
# ref: http://cmp.felk.cvut.cz/~tylecr1/facade/
# dataset download link: https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH, IMG_HEIGHT = 256, 256

il = ImageLoader(_URL, 'facades', img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
PATH = il.PATH

# 2. Input Pipeline
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
print (train_dataset)
train_dataset = train_dataset.map(il.load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(il.load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# 3. Build the models, losses and optimizers

# 3-1. Build the G model
#  Modified U-Net
generator = Generator().generate()

display_on = False
if display_on:
    inp, re = il.load(PATH + 'train/100.jpg')
    gen_output = generator(inp[tf.newaxis,...], training=False)
    plt.imshow(gen_output[0,...])
    ret = input()

# 3-2. Build teh D model
discriminator = Discriminator().generate()

if display_on:
    disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
    plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    ret = input()


# prepare the losses
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gen_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

    return LAMBDA * l1_loss + gen_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss


# prepare the optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# set the checkpoints (object-based saving)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Train and test the designed Model
EPOCHS = 5

@tf.function
def train_step(input_image, target):
    # calculate propagations and losses
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # calculate gradients
    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    # apply gradients
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                           generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                               discriminator.trainable_variables))


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # conversion to [0, 1]
        plt.axis('off')
    plt.show()


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        # training
        for input_image, target in train_ds:
            train_step(input_image, target)

        clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)

        if (epoch+1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch, time.time()-start))


fit(train_dataset, EPOCHS, test_dataset)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# run the trained model on the entire test dataset
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar)

