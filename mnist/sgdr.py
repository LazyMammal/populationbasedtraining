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

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, validation')

    sgrd(dataset, args.popsize, args.train_time, args.steps)

    print('# total time %3.1f' % main_time.elapsed())


def sgrd(dataset, popsize, train_time, training_steps, test_size=1000):
    init_op = tf.get_collection('init_op')[0]
    loss_fn = tf.get_collection('loss_fn')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss_fn)

    worker_time = Timer()
    with tf.Session() as sess:
        for worker in range(popsize):
            lr = 0.1 / (2**(worker))
            epochs = 1
            step = 0
            for _ in range(1, training_steps + 1):
                for epoch in range(epochs):
                    step += 1
                    dx = np.pi * float(epoch) / float(epochs+1)
                    learn_rate = lr * 0.5 * (1.0 + np.cos(dx))
                    batch_size = 100
                    sess.run(init_op)
                    print('%d, ' % step, end='')
                    print('%d, ' % worker, end='')
                    trainscore, testscore = workers_mod.train_graph(
                        sess, train_time, batch_size, test_size, learn_rate, dataset, train_step=train_step)
                epochs *= 2
                print('# warm restart, %3.1fs total' % worker_time.elapsed())
        print('# worker time %3.1fs' % worker_time.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?',
                        default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?',
                        default="softmax", help="tensorflow loss")
    parser.add_argument('--popsize', nargs='?', type=int,
                        default=1, help="number of workers (1)")
    parser.add_argument('--train_time', nargs='?', type=float,
                        default=1.0, help="training time per worker per step (1.0s)")
    parser.add_argument('--steps', nargs='?', type=int,
                        default=10, help="number of training steps (10)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'],
                        default='mnist', help='name of dataset')
    main(parser.parse_args())
