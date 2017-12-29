from __future__ import print_function
# pylint: disable=invalid-name

import tensorflow as tf


def make_model(x, y_):
    output_size = int(y_.get_shape()[-1])
    layer = tf.layers.dense(inputs=x, units=output_size)
    return layer
