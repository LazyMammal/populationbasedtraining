from __future__ import print_function

import sys
import argparse
from importlib import import_module

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from timer import Timer


def main(args):
    main_time = Timer()
    dataset = get_dataset(args.dataset)
    model = gen_model(args.model, args.loss)
    run_session(args, dataset, *model)
    print('total time %g' % main_time.elapsed())


def get_dataset(dataset):
    if dataset == 'mnist':
        mnist = input_data.read_data_sets('input_data/', one_hot=True)
    elif dataset == 'fashion':
        mnist = input_data.read_data_sets(
            'input_data/fashion', one_hot=True, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    return mnist


def gen_model(modelmodule, lossmodule):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    model = import_module(modelmodule)
    loss = import_module(lossmodule)

    y = model.make_model(x, y_)
    loss_fn = loss.make_loss(y, y_)

    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss_fn)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, train_step, learning_rate, accuracy


def iterate_training(sess, batch_iterations, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate):
    for i in range(batch_iterations):
        batch_xs, batch_ys = dataset.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={
            x: batch_xs, y_: batch_ys, learning_rate: learn_rate})


def run_session(args, dataset, x, y_, train_step, learning_rate, accuracy):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_session(sess, args.iterations, args.batch_size, args.learning_rate,
                      dataset, x, y_, train_step, learning_rate, accuracy)
        test_session(sess, args.batch_size, dataset, x, y_, accuracy)


def train_session(sess, iterations, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate, accuracy):
    batch_time = Timer()
    batch_iterations = 100
    for i in range(iterations // batch_iterations):
        batch_xs, batch_ys = dataset.train.next_batch(batch_size)
        train_accuracy = sess.run(
            accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        print('step %d, ' % i, end='')
        print('batch time %g, ' % batch_time.split(), end='')
        print('learning rate %g, ' % learn_rate, end='')
        print('training accuracy %g' % train_accuracy)
        iterate_training(sess, batch_iterations, batch_size, learn_rate,
                         dataset, x, y_, train_step, learning_rate)


def test_session(sess, batch_size, dataset, x, y_, accuracy):
    test_size = len(dataset.test.labels)
    print('test cases %d, ' % test_size, end='')
    acc = []
    count = 0
    for _ in range(int(test_size / batch_size)):
        batch_xs, batch_ys = dataset.test.next_batch(batch_size, shuffle=False)
        acc.append(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
        count += len(batch_ys)
    print('test accuracy %g (%d)' % (np.mean(acc), count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?',
                        default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?',
                        default="softmax", help="tensorflow loss")
    parser.add_argument('--iterations', nargs='?', type=int,
                        default=1000, help="training iterations")
    parser.add_argument('--batch_size', nargs='?', type=int,
                        default=100, help="batch size (100)")
    parser.add_argument('--learning_rate', nargs='?', type=float,
                        default=0.01, help="learning rate (0.01)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'],
                        default='mnist', help='The name of dataset')
    main(parser.parse_args())
