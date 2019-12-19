# ML for Binary Classification of IMDB reviews into (1) positive or (2) negative

from __future__ import absolute_import, division, print_function, unicode_literals

# import libs
import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import input files
(train_data, test_data), info = tfds.load(
    # Use the ver. pre-encoded w/ an ~8k vocabs
    'imdb_reviews/subwords8k',
    # return the train/test datasets as a tuple
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # return (feature, label) pairs
    as_supervised=True,
    # return the 'info structure
    with_info=True)
# print(info)

# prepare the data for training
encoder = info.features['text'].encoder
# print('Vocabulary size: {}'.format(encoder.vocab_size))

# test out the encoder
_testEncoder = False
if _testEncoder:

    sample_string = 'Hello TensorFlow.'
    encoded_string = encoder.encode(sample_string)
    print(' Encoded String: {}'.format(encoded_string))

    decoded_string = encoder.decode(encoded_string)
    print(' Decoded String: {}'.format(decoded_string))

    for ts in encoded_string:
        print(' {} ----> {}'.format(ts, encoder.decode([ts])))

    assert sample_string == decoded_string

_displayData = True
if _displayData:
    for train_example, train_label in train_data.take(1):
        print('Encoded text: {} ...'.format(train_example[:10].numpy()))
        print('Label: ', train_label.numpy())
        print('Decoded text: {} ...'.format(encoder.decode(train_example[:10])))

BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes))

test_batches = (
    test_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, test_data.output_shapes))

# for example_batch, label_batch in train_batches.take(2):
#     print('Batch shape: ', example_batch.shape)
#     print('Label shape: ', label_batch.shape)

# set the model
# # Let's build a "Continuous bag of words" style model!
# # check out the word embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation='sigmoid')])

model.summary()

# compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# train the model
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30)

# test out the trained model
loss, accuracy = model.evaluate(test_batches)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# display the result figures
history_dict = history.history
acc, val_acc = history_dict['accuracy'], history_dict['val_accuracy']
loss, val_loss = history_dict['loss'], history_dict['val_loss']
epochs_acc = range(1, len(acc)+1)
epochs_loss = range(1, len(loss)+1)

plt.figure(1)

plt.subplot(2,1,1)
plt.plot(epochs_acc, loss, 'ro', label='Training loss')
plt.plot(epochs_acc, val_loss, 'r', label='Validation loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs_acc, acc, 'bo', label='Training accuracy')
plt.plot(epochs_acc, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc.')
plt.legend()

plt.subplots_adjust(hspace=.5)
plt.show()

