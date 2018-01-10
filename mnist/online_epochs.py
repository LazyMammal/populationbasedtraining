from __future__ import print_function

import argparse
import tensorflow as tf
from timer import Timer
import mnist
import pbt
import overfit_score
import hparams as hp
import test_accuracy
import train_graph
from optimizer import get_optimizer


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, validation')
    workers = build_workers(args.popsize, [hp.resample_learnrate], [hp.perturb_learnrate])
    train_workers(dataset, workers, args.epochs, args.steps, args.cutoff, args.opt)
    print('# total time %3.1f' % main_time.elapsed())


def build_workers(popsize, hparams_fun=None, perturb_fun=None):
    build_time = Timer()
    workers = []
    for i in range(popsize):
        hparams = [fun() for fun in hparams_fun]
        worker = {'id': i, 'score': 0.0, 'score_value': 0.5,
                  'hparams': hparams, 'resample': hparams_fun, 'perturb': perturb_fun}
        workers.append(worker)

    print('# total setup time %3.1f' % build_time.elapsed())
    return workers


def train_epochs(sess, epochs, learn_rate, dataset, train_step):
    numsamples = len(dataset.train.labels)
    batch_size = 100
    iterations = epochs * numsamples // batch_size // 4
    batch_time = Timer()
    for b in range(iterations):
        train_graph.train_batch(sess, batch_size, learn_rate, dataset, train_step)
    print('%d, %f, %d, ' % (iterations * batch_size, batch_time.split(), iterations), end='')
    print('%g, ' % learn_rate, end='')
    print('%d, ' % batch_size, end='')


def train_workers(dataset, workers, epochs, training_steps, cutoff, optimizer, test_size=1000):
    train_step, init_op, reset_opt = get_optimizer(optimizer)
    step = 0
    with tf.Session() as sess:
        sess.run(init_op)
        step_time = Timer()
        for pbt_step in range(1, training_steps + 1):
            for worker in workers:
                step += 1
                print('%d, ' % step, end='')
                print('%d, ' % worker['id'], end='')
                score_value = worker['score_value']
                train_epochs(sess, epochs, worker['hparams'][0], dataset, train_step)
                train, test, valid = test_accuracy.test_graph(sess, test_size, dataset)
                print('%f, %f, %f' % (train, test, valid))
                worker['score_value'] = overfit_score.overfit_blended(train, test)
                worker['score'] = (1.0 + worker['score_value']) / (1.0 + score_value)
                pbt.tournament_replace(worker, workers, cutoff, dup_all=False, explore_fun=pbt.perturb_hparams)
            #pbt.pbt(workers, cutoff, dup_all=False)
            print('# step time %3.1fs, ' % step_time.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?', default="softmax", help="tensorflow loss")
    parser.add_argument(
        '--opt', type=str, choices=['sgd', 'momentum', 'rmsprop', 'adam'],
        default='momentum', help='optimizer (momentum)')
    parser.add_argument('--popsize', nargs='?', type=int, default=10, help="number of workers (10)")
    parser.add_argument('--cutoff', nargs='?', type=float, default=0.5, help="tournament cutoff for replacement (0.5)")
    parser.add_argument('--steps', nargs='?', type=int, default=10, help="number of training steps (10)")
    parser.add_argument('--epochs', nargs='?', type=int, default=1, help="number of epochs to train per step (1)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'], default='mnist', help='name of dataset')
    main(parser.parse_args())
