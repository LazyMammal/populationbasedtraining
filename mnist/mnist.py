from __future__ import print_function

import sys
import argparse
from importlib import import_module

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main(args):
    if args.dataset == 'mnist':
        mnist = input_data.read_data_sets('input_data/', one_hot=True)
    elif args.dataset == 'fashion':
        mnist = input_data.read_data_sets(
            'input_data/fashion', one_hot=True, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    model = import_module(args.model)
    loss = import_module(args.loss)

    y = model.make_model(x, y_)
    loss_fn = loss.make_loss(y, y_)

    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss_fn)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(args.iterations):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            if i % 100 == 0:
                train_accuracy = sess.run(
                    accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                print('step %d, ' % i, end='')
                print('learning rate %g, ' % args.learning_rate, end='')
                print('training accuracy %g' % train_accuracy)
            sess.run(train_step, feed_dict={
                     x: batch_xs, y_: batch_ys, learning_rate: args.learning_rate})

        test_size = 10000
        acc = []
        for _ in range(int(test_size / args.batch_size)):
            batch_xs, batch_ys = mnist.test.next_batch(
                args.batch_size, shuffle=False)
            acc.append(sess.run(accuracy, feed_dict={
                       x: batch_xs, y_: batch_ys}))
        print('test accuracy %g' % np.mean(acc))


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
