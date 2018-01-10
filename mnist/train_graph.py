import tensorflow as tf


def train_batch(sess, batch_size, learn_rate, dataset, train_step=None):
    if train_step is None:
        train_step = tf.get_collection('train_step')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    iterate_training(sess, 1, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate)


def iterate_training(sess, batch_iterations, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate):
    for i in range(batch_iterations):
        batch_xs, batch_ys = dataset.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: learn_rate})
