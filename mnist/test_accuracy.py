import numpy as np
import tensorflow as tf


def test_accuracy(sess, dataset, dataset_size, batch_size, x, y_, accuracy, shuffle=False):
    dropout_collection = tf.get_collection('dropout_bool')
    scores = []
    for _ in range(dataset_size // batch_size):
        batch_xs, batch_ys = dataset.next_batch(batch_size, shuffle=shuffle)
        if dropout_collection:
            dropout_bool = dropout_collection[0]
            scores.append(sess.run(accuracy, feed_dict={
                          x: batch_xs, y_: batch_ys, dropout_bool: False}))
        else:
            scores.append(sess.run(accuracy, feed_dict={
                          x: batch_xs, y_: batch_ys}))
    return np.mean(scores)


def test_graph(sess, batch_size, dataset):
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    accuracy = tf.get_collection('accuracy')[0]

    testdata_size = len(dataset.test.labels)
    trainscore = test_accuracy(sess, dataset.train, testdata_size, batch_size, x, y_, accuracy, True)
    testscore = test_accuracy(sess, dataset.test, testdata_size, batch_size, x, y_, accuracy)
    validscore = test_accuracy(sess, dataset.validation, testdata_size, batch_size, x, y_, accuracy)

    return (trainscore, testscore, validscore)
