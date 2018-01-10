from __future__ import print_function

import argparse
import tensorflow as tf
from timer import Timer
import mnist
import test_accuracy
import train_graph
import hparams as hp
import workers as workers_mod
from optimizer import get_optimizer


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, validation')

    search_grid_epochs(dataset, args.steps, args.learnrate, args.opt, args.workerid)
    #search_grid(dataset, args.popsize, args.train_time, args.steps)
    #multi_random(dataset, args.popsize, args.train_time, args.steps)

    print('# total time %3.1f' % main_time.elapsed())


def search_grid_epochs(dataset, epochs, learnlist=[0.1], optimizer='sgd', start_wid=0, test_size=1000):
    train_step, init_op, reset_opt = get_optimizer(optimizer)
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
        train_graph.train_batch(sess, batch_size, learn_rate, dataset, train_step)
    print('%d, %f, %d, ' % (iterations * batch_size, batch_time.split(), iterations), end='')
    print('%g, ' % learn_rate, end='')
    print('%d, ' % batch_size, end='')
    print('%f, %f, %f' % test_accuracy.test_graph(sess, test_size, dataset))
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
        '--opt', type=str, choices=['sgd', 'momentum', 'rmsprop', 'adam'],
        default='momentum', help='optimizer (momentum)')
    parser.add_argument('--popsize', nargs='?', type=int, default=1, help="number of workers (1)")
    parser.add_argument('--workerid', nargs='?', type=int, default=0, help="starting worker id number (0)")
    parser.add_argument('--steps', nargs='?', type=int, default=1, help="number of training steps (1)")
    parser.add_argument('--learnrate', nargs='*', type=float, default=[0.1], help="learning rate (0.1)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
