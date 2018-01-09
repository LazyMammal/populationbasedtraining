from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf
from timer import Timer
import mnist
from test_accuracy import test_accuracy
import hparams as hp
import workers as workers_mod


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, learnrate, learnrate, batchsize, trainaccuracy, testaccuracy, validation')
    sgdr(dataset, args.popsize, args.steps)
    print('# total time %3.1f' % main_time.elapsed())


def sgdr(dataset, popsize, training_steps, test_size=1000):
    loss_fn = tf.get_collection('loss_fn')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_fn)
    # train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss_fn)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss_fn)
    init_op = tf.global_variables_initializer()

    worker_time = Timer()
    with tf.Session() as sess:
        for wid in range(popsize):
            sess.run(init_op)
            learn_rate = 0.8 / (2**wid)
            epochs = 1
            step = 0
            for _ in range(1, training_steps + 1):
                step = train_restart(sess, wid, epochs, step, learn_rate, dataset, test_size, train_step)
                epochs *= 2
                print('# warm restart, %3.1fs total' % worker_time.elapsed())
        print('# worker time %3.1fs' % worker_time.split())


def train_restart(sess, wid, epochs, step, learn_rate, dataset, test_size, train_step):
    numsamples = len(dataset.train.labels)
    for epoch in range(epochs):
        step += 1
        print('%d, ' % step, end='')
        print('%d, ' % wid, end='')
        batch_size = 100
        iterations = numsamples // batch_size
        batch_time = Timer()
        lr_start = scale_learn_rate(learn_rate, epoch, epochs, 0, iterations)
        for b in range(iterations):
            lr = scale_learn_rate(learn_rate, epoch, epochs, b, iterations)
            train_batch(sess, batch_size, lr, dataset, train_step)
        print('%d, %f, %d, ' % (numsamples, batch_time.split(), iterations), end='')
        print('%g, ' % lr_start, end='')
        print('%d, ' % batch_size, end='')
        test_graph(sess, batch_size, test_size, dataset)
    return step


def test_learn_rate():
    epochs = 4
    learn_rate = .1
    for epoch in range(epochs):
        iterations = 4
        print(epoch)
        for b in range(iterations):
            lr = scale_learn_rate(learn_rate, epoch, epochs, b, iterations)
            print(lr, learn_rate, epoch, epochs, b, iterations)


def scale_learn_rate(learn_rate, epoch, epochs, b, iterations):
    dx = (epoch * iterations + b) / float(epochs * iterations)
    lr = learn_rate * 0.5 * (1.0 + np.cos(dx * np.pi))
    return lr


def train_batch(sess, batch_size, learn_rate, dataset, train_step=None):
    if train_step is None:
        train_step = tf.get_collection('train_step')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    learning_rate = tf.get_collection('learning_rate')[0]

    mnist.iterate_training(sess, 1, batch_size, learn_rate, dataset, x, y_, train_step, learning_rate)


def test_graph(sess, batch_size, test_size, dataset):
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    accuracy = tf.get_collection('accuracy')[0]

    testdata_size = len(dataset.test.labels)
    trainscore = test_accuracy(sess, dataset.train, testdata_size, test_size, x, y_, accuracy, True)
    testscore = test_accuracy(sess, dataset.test, testdata_size, test_size, x, y_, accuracy)
    validscore = test_accuracy(sess, dataset.validation, testdata_size, test_size, x, y_, accuracy)

    print('%f, ' % trainscore, end='')
    print('%f, ' % testscore, end='')
    print('%f' % validscore)

    return (trainscore, testscore)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument('--popsize', nargs='?', type=int, default=1, help="number of workers (1)")
    parser.add_argument('--steps', nargs='?', type=int, default=10, help="number of training steps (10)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
