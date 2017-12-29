from __future__ import print_function
# pylint: disable=invalid-name

import tensorflow as tf


def make_model(x, y_):
    input_size = int(x.get_shape()[-1])
    output_size = int(y_.get_shape()[-1])
    hidden_size = int((input_size + output_size)/2)

    L1 = tf.layers.dense(inputs=x, units=hidden_size, activation=tf.nn.tanh)
    L2 = tf.layers.dense(inputs=L1, units=output_size)

    return L2
