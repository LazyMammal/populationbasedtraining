from __future__ import print_function

import numpy as np
import tensorflow as tf
import mnist
from timer import Timer


def main():
    feed_dict()


def feed_dict():
    dataset = mnist.get_dataset('fashion')
    x, y_, train_step, learning_rate, accuracy = mnist.gen_model('conv_dropout_model', 'softmax')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        learn_rate = 0.01
        datasize = len(dataset.train.labels) // 4
        for batch_size in [1, 2, 4, 8, 16, 32, 64] + list(range(100, 3000, 100)):
            epoch_time = Timer()
            iterations = datasize // batch_size
            for i in range(iterations):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: learn_rate})
            split = epoch_time.split()
            print('%d, %d, %3.1fs, %d/s' % (batch_size, iterations, split, datasize // split))


if __name__ == '__main__':
    main()
