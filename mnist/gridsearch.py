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

    #search_grid(dataset, args.popsize, args.train_time, args.steps)
    multi_random(dataset, args.popsize, args.train_time, args.steps)

    print('# total time %3.1f' % main_time.elapsed())


def multi_random(dataset, popsize, train_time, training_steps, test_size=1000):
    init_op = tf.get_collection('init_op')[0]
    worker_time = Timer()
    with tf.Session() as sess:
        for worker in range(popsize):
            learn_rate = hp.resample_learnrate()
            batch_size = hp.resample_batchsize()
            sess.run(init_op)
            for step in range(1, training_steps + 1):
                print('%d, ' % step, end='')
                print('%d, ' % worker, end='')
                workers_mod.train_graph(sess, dataset, train_time, test_size,
                                        batch_size, learn_rate)
            print('# worker time %3.1fs' % worker_time.split())


def search_grid(dataset, popsize, train_time, training_steps, test_size=1000):
    init_op = tf.get_collection('init_op')[0]
    worker = 0
    with tf.Session() as sess:
        for batch_size in [2**b for b in range(4, 11)]:
            for learn_rate in [2**r for r in range(-3, -22, -3)]:
                sess.run(init_op)
                for step in range(1, training_steps + 1):
                    print('%d, ' % step, end='')
                    print('%d, ' % worker, end='')
                    workers_mod.train_graph(sess, dataset, train_time, test_size,
                                            batch_size, learn_rate)
                worker += 1


if __name__ == '__main__':
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
