from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf
from timer import Timer
import mnist
import pbt
from test_accuracy import test_accuracy
from overfit_score import overfit_score
import hparams as hp
import workers as workers_mod


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, time')

    workers = build_workers(args.popsize,
                            [hp.resample_learnrate, hp.resample_batchsize],
                            [hp.perturb_learnrate, hp.perturb_batchsize])

    train_workers(dataset, workers, args.train_time,
                  args.steps, test_size=1000)

    print('# total time %3.1f' % main_time.elapsed())


def build_workers(popsize, hparams_fun=None, perturb_fun=None):
    build_time = Timer()
    workers = []
    for i in range(popsize):
        hparams = [fun() for fun in hparams_fun]
        worker = {'id': i, 'score': 0.0,
                  'hparams': hparams, 'resample': hparams_fun, 'perturb': perturb_fun}
        workers.append(worker)

    print('# total setup time %3.1f' % build_time.elapsed())
    return workers


def train_workers(dataset, workers, train_time, training_steps, test_size=1000):
    init_op = tf.get_collection('init_op')[0]
    worker = 0
    with tf.Session() as sess:
        sess.run(init_op)
        step_time = Timer()
        for step in range(1, training_steps + 1):
            for worker in workers:
                print('%d, ' % step, end='')
                print('%d, ' % worker['id'], end='')
                score = workers_mod.train_graph(sess, train_time,
                                                worker['hparams'][1], test_size,
                                                worker['hparams'][0], dataset)
                worker['score'] = 1.0 - overfit_score(*score)
            pbt.pbt(workers, dup_all=False)
            print('# step time %3.1fs, ' % step_time.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?',
                        default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?',
                        default="softmax", help="tensorflow loss")
    parser.add_argument('--popsize', nargs='?', type=int,
                        default=10, help="number of workers (10)")
    parser.add_argument('--train_time', nargs='?', type=float,
                        default=1.0, help="training time per worker per step (1.0s)")
    parser.add_argument('--steps', nargs='?', type=int,
                        default=100, help="number of training steps (100)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'],
                        default='mnist', help='name of dataset')
    main(parser.parse_args())
