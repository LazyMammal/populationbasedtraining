from __future__ import print_function
# pylint: disable=invalid-name

import tensorflow as tf


def make_model(x, y_):
    input_size = int(x.get_shape()[-1])
    output_size = int(y_.get_shape()[-1])
    hidden_size = (input_size + output_size)//2

    W1 = tf.Variable(tf.random_uniform(shape=[input_size, hidden_size], dtype=tf.float32))
    L1 = tf.tanh( tf.matmul(x, W1) )

    W2 = tf.Variable(tf.random_uniform(shape=[hidden_size, output_size], dtype=tf.float32))
    L2 = tf.matmul(L1, W2)

    return L2
