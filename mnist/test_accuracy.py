import numpy as np
import tensorflow as tf


def test_accuracy(sess, dataset, test_size, batch_size, x, y_, accuracy, shuffle=False):
    dropout_collection = tf.get_collection('dropout_bool')
    scores = []
    for _ in range(test_size // batch_size):
        batch_xs, batch_ys = dataset.next_batch(batch_size, shuffle=shuffle)
        if dropout_collection:
            dropout_bool = dropout_collection[0]
            scores.append(sess.run(accuracy, feed_dict={
                          x: batch_xs, y_: batch_ys, dropout_bool: False}))
        else:
            scores.append(sess.run(accuracy, feed_dict={
                          x: batch_xs, y_: batch_ys}))
    return np.mean(scores)
