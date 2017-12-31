from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf
from timer import Timer
import mnist
import pbt


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    workers = build_workers(args.popsize, dataset,
                            [resample_learnrate, resample_batchsize],
                            [perturb_learnrate, perturb_batchsize])

    train_workers(workers, args.train_time, args.steps, pbt.pbt)

    print('total time %3.1f' % main_time.elapsed())


def build_workers(popsize, dataset, hparams_fun=None, perturb_fun=None):
    build_time = Timer()
    init_op = tf.get_collection('init_op')[0]

    saver = tf.train.Saver(max_to_keep=popsize)

    with tf.Session() as sess:
        workers = []
        for i in range(popsize):
            sess.run(init_op)
            name = 'ckpt/worker_' + str(i) + '.ckpt'
            saver.save(sess, name)
            hparams = [fun() for fun in hparams_fun]
            worker = {'name': name, 'dup_from_name': None, 'id': i, 'score': 0.0,
                      'hparams': hparams, 'resample': hparams_fun, 'perturb': perturb_fun,
                      'dataset': dataset}
            workers.append(worker)

            print('worker (%d) setup time %3.1f' % (i, build_time.split()))
        print('total setup time %3.1f' % build_time.elapsed())
    sess.close()
    return workers


def resample_learnrate():
    return 2.0**np.log(np.random.lognormal() / 10.0) / 100.0


def resample_batchsize():
    return int(np.random.logseries(.95) * 10)


def perturb_learnrate(learnrate):
    return pbt.perturb(learnrate, 0.0, 0.1)


def perturb_batchsize(batchsize):
    return int(pbt.perturb(batchsize / 10.0, 1, 100) * 10)


def train_workers(workers, train_time, training_steps, step_callback=None, test_size=1000):
    step_time = Timer()
    for step in range(1, training_steps + 1):
        io_accum = 0.0
        for worker in workers:
            print('step %d, ' % step, end='')
            io_accum += train_worker(worker, train_time, test_size)
        if step_callback:
            step_callback(workers)
        print('step time %3.1fs, ' % step_time.split(), end='')
        print('io time %3.1fs' % io_accum)


def train_worker(worker, train_time, test_size):
    io_time = Timer()
    io_accum = 0.0
    tf.reset_default_graph()
    with tf.Session() as sess:
        io_time.split()
        name = worker['dup_from_name'] or worker['name']
        saver2 = tf.train.import_meta_graph(name + '.meta')
        saver2.restore(sess, name)
        worker['dup_from_name'] = None
        io_accum += io_time.split()
        print('worker %d, ' % worker['id'], end='')
        score = train_graph(sess, train_time, worker['hparams'][1],
                            test_size, worker['hparams'][0], worker['dataset'])
        worker['score'] = score
        io_time.split()
        saver2.save(sess, worker['name'])
        io_accum += io_time.split()
    return io_accum


def train_graph(sess, train_time, batch_size, test_size, learn_rate, dataset):
    train_step = tf.get_collection('train_step')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    accuracy = tf.get_collection('accuracy')[0]

    batch_time = Timer()
    iterations = 50
    total_iterations = 0
    count = 0
    while batch_time.elapsed() < train_time:
        mnist.iterate_training(sess, iterations, batch_size, learn_rate,
                               dataset, x, y_, train_step, learning_rate)
        count += 1
        total_iterations += iterations

    print('samples trained %d (%3.1fs) %d, ' %
          (total_iterations * batch_size, batch_time.split(), count), end='')
    print('learning rate %3.3g, ' % learn_rate, end='')
    print('batch size %d, ' % batch_size, end='')

    testdata_size = len(dataset.test.labels)
    trainscore = test_accuracy(
        sess, dataset.train, testdata_size, test_size, x, y_, accuracy, True)
    testscore = test_accuracy(
        sess, dataset.test, testdata_size, test_size, x, y_, accuracy)

    print('train %3.3f, ' % trainscore, end='')
    print('test %3.3f (%3.1fs)' % (testscore, batch_time.split()))

    return testscore


def test_accuracy(sess, dataset, test_size, batch_size, x, y_, accuracy, shuffle=False):
    scores = []
    for _ in range(test_size // batch_size):
        batch_xs, batch_ys = dataset.next_batch(batch_size, shuffle=shuffle)
        scores.append(sess.run(accuracy, feed_dict={
                      x: batch_xs, y_: batch_ys}))
    return np.mean(scores)


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
