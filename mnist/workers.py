from __future__ import print_function

import sys
import argparse
from importlib import import_module

import tensorflow as tf
from timer import Timer
import mnist


def main(args):
    dataset = mnist.get_dataset(args.dataset)
    modelmodule = import_module(args.model)
    lossmodule = import_module(args.loss)
    mnist.gen_model(modelmodule, lossmodule)

    workers = build_workers(args.popsize)
    tf.reset_default_graph()
    train_workers(workers, dataset, args.train_time, args.steps)
    print('total time %3.1f' % main_time.elapsed())


def build_workers(popsize):
    init_op = tf.get_collection('init_op')[0]

    saver = tf.train.Saver(max_to_keep=popsize)

    with tf.Session() as sess:
        workers = []
        for i in range(popsize):
            sess.run(init_op)
            name = 'ckpt/worker_' + str(i) + '.ckpt'
            saver.save(sess, name)
            workers.append({'name': name, 'id': i})
            print('worker (%d) setup time %3.1f' % (i, main_time.split()))
        print('total setup time %3.1f' % main_time.elapsed())
    sess.close()
    return workers


def train_workers(workers, dataset, train_time, training_steps):
    batch_size = 100
    test_size = 1000
    learn_rate = 0.01

    with tf.Session() as sess:
        for step in range(1, training_steps + 1):
            for worker in workers:
                saver2 = tf.train.import_meta_graph(worker['name'] + '.meta')
                saver2.restore(sess, worker['name'])
                print('step %d, ' % step, end='')
                print('worker %d, ' % worker['id'], end='')
                score = train_graph(sess, train_time, batch_size,
                                    test_size, learn_rate, dataset)
                worker['score'] = score
                saver2.save(sess, worker['name'])
            print('step time %3.1f' % main_time.split())


def train_graph(sess, train_time, batch_size, test_size, learn_rate, dataset):
    train_step = tf.get_collection('train_step')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    accuracy = tf.get_collection('accuracy')[0]

    batch_time = Timer()
    batch_iterations = 100
    count = 0
    while batch_time.elapsed() < train_time:
        mnist.iterate_training(sess, batch_iterations, batch_size, learn_rate,
                               dataset, x, y_, train_step, learning_rate)
        count += batch_iterations
    batch_xs, batch_ys = dataset.train.next_batch(test_size)
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    batch_xs, batch_ys = dataset.test.next_batch(test_size)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print('batch time %3.1f (%d), ' % (batch_time.split(), count), end='')
    print('learning rate %3.3g, ' % learn_rate, end='')
    print('training accuracy %3.3f, ' % train_accuracy, end='')
    print('testing accuracy %3.3f' % test_accuracy)
    return test_accuracy


if __name__ == '__main__':
    main_time = Timer()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?',
                        default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?',
                        default="softmax", help="tensorflow loss")
    parser.add_argument('--popsize', nargs='?', type=int,
                        default=10, help="number of workers (10)")
    parser.add_argument('--train_time', nargs='?', type=float,
                        default=10.0, help="training time per worker per step (10.0s)")
    parser.add_argument('--steps', nargs='?', type=int,
                        default=10, help="number of training steps (10)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'],
                        default='mnist', help='name of dataset')
    main(parser.parse_args())
