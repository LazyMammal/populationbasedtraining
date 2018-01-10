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
import sgdr


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, validation')

    search_grid_epochs(dataset, args.popsize, args.steps, args.learnrate, args.opt, args.workerid)
    #search_grid(dataset, args.popsize, args.train_time, args.steps)
    #multi_random(dataset, args.popsize, args.train_time, args.steps)

    print('# total time %3.1f' % main_time.elapsed())


def search_grid_epochs(dataset, popsize, epochs, learnlist=[0.1], optimizer='sgd', start_wid=0, test_size=1000):
    loss_fn = tf.get_collection('loss_fn')[0]
    learning_rate = tf.get_collection('learning_rate')[0]

    print('#', optimizer, learnlist)

    if optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    else:  # 'sgd'
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    beta = 0.01
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    weights = tf.add_n([tf.nn.l2_loss(var) for var in var_list if var is not None])
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss_fn + beta * regularizer)
    train_step = opt.minimize(loss)
    init_op = tf.global_variables_initializer()

    worker_time = Timer()
    with tf.Session() as sess:
        for wid, learn_rate in enumerate(learnlist):
            step = 0
            sess.run(init_op)
            for e in range(epochs):
                step = train_epochs(sess, wid + start_wid, 1, step, learn_rate, dataset, test_size, train_step)
            print('# worker time %3.1fs' % worker_time.split())


def train_epochs(sess, wid, epochs, step, learn_rate, dataset, test_size, train_step):
    numsamples = len(dataset.train.labels)
    step += 1
    print('%d, ' % step, end='')
    print('%d, ' % wid, end='')
    batch_size = 100
    iterations = epochs * numsamples // batch_size // 4
    batch_time = Timer()
    for b in range(iterations):
        sgdr.train_batch(sess, batch_size, learn_rate, dataset, train_step)
    print('%d, %f, %d, ' % (iterations * batch_size, batch_time.split(), iterations), end='')
    print('%g, ' % learn_rate, end='')
    print('%d, ' % batch_size, end='')
    sgdr.test_graph(sess, batch_size, test_size, dataset)
    return step


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
    parser.add_argument('--model', nargs='?', default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument(
        '--opt', type=str, choices=['sgd', 'momentum', 'rmsprop'],
        default='momentum', help='optimizer (momentum)')
    parser.add_argument('--popsize', nargs='?', type=int, default=1, help="number of workers (1)")
    parser.add_argument('--workerid', nargs='?', type=int, default=0, help="starting worker id number (0)")
    parser.add_argument('--steps', nargs='?', type=int, default=1, help="number of training steps (1)")
    parser.add_argument('--learnrate', nargs='*', type=float, default=[0.1], help="learning rate (0.1)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
