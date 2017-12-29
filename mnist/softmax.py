from __future__ import print_function

import tensorflow as tf


def make_loss(y, y_):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    return cross_entropy
