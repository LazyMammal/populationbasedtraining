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
    sgdr(dataset, args.popsize, args.steps, args.learnrate, args.opt, args.workerid)
    print('# total time %3.1f' % main_time.elapsed())


def sgdr(dataset, popsize, training_steps, learnlist=[0.1], optimizer='sgd', start_wid=0, test_size=1000):
    train_step, init_op, reset_opt = get_optimizer(optimizer)
    worker_time = Timer()
    with tf.Session() as sess:
        for wid, learn_rate in zip(
                range(start_wid, start_wid + popsize),
                np.repeat(learnlist, int(0.5 + float(popsize) / len(learnlist)))):
            sess.run(init_op)
            epochs = 1
            step = 0
            for _ in range(1, training_steps + 1):
                sess.run(reset_opt)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument(
        '--opt', type=str, choices=['sgd', 'momentum', 'rmsprop', 'adam'],
        default='momentum', help='optimizer (momentum)')
    parser.add_argument('--popsize', nargs='?', type=int, default=1, help="number of workers (1)")
    parser.add_argument('--workerid', nargs='?', type=int, default=0, help="starting worker id number (0)")
    parser.add_argument('--steps', nargs='?', type=int, default=3, help="number of training steps (3)")
    parser.add_argument('--learnrate', nargs='*', type=float, default=[0.1], help="learning rate (0.1)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
