from __future__ import print_function
# pylint: disable=invalid-name

import tensorflow as tf


def make_model(x, y_):
    input_size = int(x.get_shape()[-1])
    output_size = int(y_.get_shape()[-1])
    W = tf.Variable(tf.random_uniform(shape=[input_size, output_size], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform(shape=[output_size], dtype=tf.float32))
    y = tf.matmul(x, W) + b
    return y
