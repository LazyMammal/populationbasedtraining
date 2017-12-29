from __future__ import print_function
# pylint: disable=invalid-name

import tensorflow as tf
import numpy as np


def make_model(x, y_):
    input_size = int(x.get_shape()[-1])
    input_edge = int(np.sqrt(input_size))
    output_size = int(y_.get_shape()[-1])

    # --> [batch, n, n, 1]
    input_layer = tf.reshape(x, [-1, input_edge, input_edge, 1])

    # --> [batch, n, n, 32]
    conv1 = tf.layers.conv2d(
        input_layer, 32, [5, 5], padding="same", activation=tf.nn.relu)

    # --> [batch, n/2, n/2, 32]
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2)

    '''
    # --> [batch, n/2, n/2, 16]
    conv2 = tf.layers.conv2d(
        pool1, 16, [5, 5], padding="same", activation=tf.nn.relu)

    # --> [batch, n/4, n/4, 16]
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=2)

    # --> [batch, ?]
    pool2_flat = tf.contrib.layers.flatten(pool2)
    '''
    # --> [batch, ?]
    pool2_flat = tf.contrib.layers.flatten(pool1)
    
    # --> [batch, 1024]
    dense = tf.layers.dense(pool2_flat, 1024, tf.nn.relu)

    # --> [batch, output_size]
    logits = tf.layers.dense(dense, output_size)

    return logits
