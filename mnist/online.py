from __future__ import print_function

import argparse
import tensorflow as tf
from timer import Timer
import mnist
import pbt
import overfit_score
import hparams as hp
import workers as workers_mod
from optimizer import get_optimizer


def main(args):
    main_time = Timer()
    dataset = mnist.get_dataset(args.dataset)
    mnist.gen_model(args.model, args.loss)

    print('step, worker, samples, time, loops, learnrate, batchsize, trainaccuracy, testaccuracy, validation')

    workers = build_workers(args.popsize,
                            [hp.resample_learnrate, hp.resample_batchsize],
                            [hp.perturb_learnrate, hp.perturb_batchsize])

    train_workers(dataset, workers, args.train_time,
                  args.steps, args.cutoff, test_size=1000)

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


def train_workers(dataset, workers, train_time, training_steps, cutoff, test_size=1000):
    train_step, init_op, reset_opt = get_optimizer()
    worker = 0
    with tf.Session() as sess:
        sess.run(init_op)
        step_time = Timer()
        for step in range(1, training_steps + 1):
            for worker in workers:
                print('%d, ' % step, end='')
                print('%d, ' % worker['id'], end='')
                score_value = worker['score_value']
                time_available = train_time * (0.1 + worker['score'])
                trainscore, testscore = workers_mod.train_graph(sess, time_available,
                                                                worker['hparams'][1], test_size,
                                                                worker['hparams'][0], dataset, train_step=train_step)
                worker['score_value'] = overfit_score.overfit_blended(
                    trainscore, testscore)
                worker['score'] = (1.0 + worker['score_value']) / (1.0 + score_value)
                pbt.tournament_replace(worker, workers, cutoff, dup_all=False, explore_fun=pbt.perturb_hparams)
            #pbt.pbt(workers, cutoff, dup_all=False)
            print('# step time %3.1fs, ' % step_time.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?',
                        default="bias_layer", help="tensorflow model")
    parser.add_argument('--loss', nargs='?',
                        default="softmax", help="tensorflow loss")
    parser.add_argument('--popsize', nargs='?', type=int,
                        default=10, help="number of workers (10)")
    parser.add_argument('--cutoff', nargs='?', type=float,
                        default=0.2, help="fraction of population to replace after each step (0.2)")
    parser.add_argument('--train_time', nargs='?', type=float,
                        default=1.0, help="training time per worker per step (1.0s)")
    parser.add_argument('--steps', nargs='?', type=int,
                        default=100, help="number of training steps (100)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion'],
                        default='mnist', help='name of dataset')
    main(parser.parse_args())
