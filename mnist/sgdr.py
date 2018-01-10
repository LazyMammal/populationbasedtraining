from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from timer import Timer
import mnist
import test_accuracy
import train_graph
from optimizer import get_optimizer


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, learnrate, learnrate, batchsize, trainaccuracy, testaccuracy, validation')
    sgdr(dataset, args.popsize, args.epochs, args.learnrate, args.epochmult, args.epochmin, args.opt, args.workerid)
    print('# total time %3.1f' % main_time.elapsed())


def sgdr(dataset, popsize, training_steps, learnlist=[0.1], epochlist=[2.0], minlist=[1.0], optimizer='sgd', start_wid=0, test_size=1000):
    train_step, init_op, reset_opt = get_optimizer(optimizer)
    worker_time = Timer()
    with tf.Session() as sess:
        popsize = max(popsize, len(learnlist), len(epochlist), len(minlist))
        for wid, learn_rate, epochmult, epochmin in zip(
                range(start_wid, start_wid + popsize),
                np.repeat(learnlist, int(0.5 + float(popsize) / len(learnlist))),
                np.repeat(epochlist, int(0.5 + float(popsize) / len(epochlist))),
                np.repeat(minlist, int(0.5 + float(popsize) / len(minlist)))
        ):
            sess.run(init_op)
            epochs = epochmin
            step = 0
            while step < training_steps:
                print('#', optimizer, ', lr', learn_rate, 'epochs', epochs, 'mult', epochmult)
                sess.run(reset_opt)
                iterations = min(training_steps - step, int(epochs + 0.5))
                step = train_restart(sess, wid, iterations, step, learn_rate, dataset, test_size, train_step)
                epochs *= epochmult
                print('# warm restart, %3.1fs total' % worker_time.elapsed())
            print('# worker time %3.1fs' % worker_time.split())


def train_restart(sess, wid, epochs, step, learn_rate, dataset, test_size, train_step):
    numsamples = len(dataset.train.labels)
    for epoch in range(epochs):
        step += 1
        print('%d, ' % step, end='')
        print('%d, ' % wid, end='')
        batch_size = 100
        iterations = numsamples // batch_size // 4
        batch_time = Timer()
        lr_start = scale_learn_rate(learn_rate, epoch, epochs, 0, iterations)
        for b in range(iterations):
            lr = scale_learn_rate(learn_rate, epoch, epochs, b, iterations)
            train_graph.train_batch(sess, batch_size, lr, dataset, train_step)
        print('%d, %f, %d, ' % (iterations * batch_size, batch_time.split(), iterations), end='')
        print('%g, ' % lr_start, end='')
        print('%d, ' % batch_size, end='')
        print('%f, %f, %f' % test_accuracy.test_graph(sess, test_size, dataset))
    return step


def scale_learn_rate(learn_rate, epoch, epochs, b, iterations):
    dx = (epoch * iterations + b) / float(epochs * iterations)
    lr = learn_rate * 0.5 * (1.0 + np.cos(dx * np.pi))
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument(
        '--opt', type=str, choices=['sgd', 'momentum', 'rmsprop', 'adam'],
        default='momentum', help='optimizer (momentum)')
    parser.add_argument('--popsize', nargs='?', type=int, default=1, help="number of workers (1)")
    parser.add_argument('--workerid', nargs='?', type=int, default=0, help="starting worker id number (0)")
    parser.add_argument('--epochs', nargs='?', type=int, default=128, help="total number of epochs to train (128)")
    parser.add_argument('--learnrate', nargs='*', type=float, default=[0.1], help="learning rate (0.1)")
    parser.add_argument('--epochmult', nargs='*', type=float, default=[2.0], help="epoch count multiplier (2.0)")
    parser.add_argument('--epochmin', nargs='*', type=float, default=[1.0], help="minimum epoch count (1.0)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
