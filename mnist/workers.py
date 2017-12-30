from __future__ import print_function

from importlib import import_module

import tensorflow as tf
from timer import Timer
import mnist


def main():
    main_time = Timer()
    dataset = mnist.get_dataset('fashion')     # mnist
    modelmodule = import_module('conv_model')  # tanh_layer
    lossmodule = import_module('softmax')
    model = mnist.gen_model(modelmodule, lossmodule)
    init_op = tf.global_variables_initializer()

    popsize = 5
    saver = tf.train.Saver(max_to_keep=popsize)

    batch_size = 100
    test_size = 1000
    learn_rate = 0.01

    with tf.Session() as sess:
        workers = []
        for i in range(popsize):
            sess.run(init_op)
            name = 'ckpt/worker_' + str(i) + '.ckpt'
            saver.save(sess, name)
            workers.append(name)
            print('worker (%d) setup time %3.1f' % (i, main_time.split()))
        print('total setup time %3.1f' % main_time.elapsed())

        for step in range(5):
            for wid, name in enumerate(workers):
                saver.restore(sess, name)
                print('step %d, ' % step, end='')
                print('worker %d, ' % wid, end='')
                train_graph(sess, 3.0, batch_size, test_size, learn_rate, dataset, *model)
                saver.save(sess, name)
            print('step time %3.1f' % main_time.split())

        print('total time %3.1f' % main_time.elapsed())


def train_graph(sess, train_time, batch_size, test_size, learn_rate, dataset, x, y_, train_step, learning_rate, accuracy):
    batch_time = Timer()
    batch_iterations = 10
    count = 0
    while batch_time.elapsed() < train_time:
        mnist.iterate_training(sess, batch_iterations, batch_size, learn_rate,
                               dataset, x, y_, train_step, learning_rate)
        count += 1
    batch_xs, batch_ys = dataset.train.next_batch(test_size)
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    batch_xs, batch_ys = dataset.test.next_batch(test_size)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print('batch time %3.1f (%d), ' % (batch_time.split(), count), end='')
    print('learning rate %3.3g, ' % learn_rate, end='')
    print('training accuracy %3.3f, ' % train_accuracy, end='')
    print('testing accuracy %3.3f' % test_accuracy)


if __name__ == '__main__':
    main()
