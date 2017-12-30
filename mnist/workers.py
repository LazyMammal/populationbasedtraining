from __future__ import print_function

import sys
import argparse

import numpy as np
import tensorflow as tf
from timer import Timer
import mnist


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    workers = build_workers(args.popsize, dataset, [
                            resample_learnrate, resample_batchsize])
    tf.reset_default_graph()
    train_workers(workers, args.train_time, args.steps)
    print('total time %3.1f' % main_time.elapsed())


def build_workers(popsize, dataset, hparams_fun=None):
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
                      'hparams': hparams, 'resample': hparams_fun, 'dataset': dataset}
            workers.append(worker)

            print('worker (%d) setup time %3.1f' % (i, build_time.split()))
        print('total setup time %3.1f' % build_time.elapsed())
    sess.close()
    return workers


def resample_learnrate():
    return 2.0**np.log(np.random.lognormal() / 10.0) / 100.0


def resample_batchsize():
    return int(np.random.logseries(.95) * 10)


def train_workers(workers, train_time, training_steps, test_size=1000):
    step_time = Timer()
    with tf.Session() as sess:
        for step in range(1, training_steps + 1):
            for worker in workers:
                name = worker['dup_from_name'] or worker['name']
                saver2 = tf.train.import_meta_graph(name + '.meta')
                saver2.restore(sess, name)
                worker['dup_from_name'] = None
                print('step %d, ' % step, end='')
                print('worker %d, ' % worker['id'], end='')
                score = train_graph(sess, train_time, worker['hparams'][1],
                                    test_size, worker['hparams'][0], worker['dataset'])
                worker['score'] = score
                saver2.save(sess, worker['name'])
            print('step time %3.1f' % step_time.split())


def train_graph(sess, train_time, batch_size, test_size, learn_rate, dataset):
    train_step = tf.get_collection('train_step')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    accuracy = tf.get_collection('accuracy')[0]

    batch_time = Timer()
    batch_iterations = 100
    count = 0
    while batch_time.elapsed() < train_time:
        mnist.iterate_training(sess, batch_iterations, batch_size, learn_rate,
                               dataset, x, y_, train_step, learning_rate)
        count += batch_iterations
    batch_xs, batch_ys = dataset.train.next_batch(test_size)
    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    batch_xs, batch_ys = dataset.test.next_batch(test_size)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print('batch time %3.1f (%d), ' % (batch_time.split(), count), end='')
    print('learning rate %3.3g, ' % learn_rate, end='')
    print('batch size %d, ' % batch_size, end='')
    print('training accuracy %3.3f, ' % train_accuracy, end='')
    print('testing accuracy %3.3f' % test_accuracy)
    return test_accuracy


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
