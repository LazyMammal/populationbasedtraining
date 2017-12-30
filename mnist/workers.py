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

    popsize = 10
    saver = tf.train.Saver(max_to_keep=popsize)

    batch_size = 100
    learn_rate = 0.01

    with tf.Session() as sess:
        workers = []
        for i in range(popsize):
            sess.run(init_op)
            name = 'ckpt/worker_' + str(i) + '.ckpt'
            saver.save(sess, name)
            workers.append(name)
            print('worker (%d) setup time %g' % (i, main_time.split()))
        print('total setup time %g' % main_time.elapsed())

        for step in range(10):
            for wid, name in enumerate(workers):
                saver.restore(sess, name)
                print('step %d, ' % step, end='')
                print('worker %d, ' % wid, end='')
                train_graph(sess, batch_size, learn_rate, dataset, *model)
                saver.save(sess, name)
            print('step time %g' % main_time.split())

        print('total time %g' % main_time.elapsed())


def train_graph(sess, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate, accuracy):
    batch_time = Timer()
    batch_iterations = 100
    mnist.iterate_training(sess, batch_iterations, batch_size, learn_rate,
                           dataset, x, y_, train_step, learning_rate)
    batch_xs, batch_ys = dataset.train.next_batch(batch_size)
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    batch_xs, batch_ys = dataset.test.next_batch(batch_size)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print('batch time %g, ' % batch_time.split(), end='')
    print('learning rate %g, ' % learn_rate, end='')
    print('training accuracy %g, ' % train_accuracy, end='')
    print('testing accuracy %g' % test_accuracy)


if __name__ == '__main__':
    main()
