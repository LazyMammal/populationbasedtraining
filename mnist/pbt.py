from __future__ import print_function

import numpy as np
import copy
from timer import Timer


def pbt(workers, cutoff=0.2):
    # pbt_time = Timer()
    truncate_pop(workers, cutoff, explore_fun=perturb_hparams)
    # print('# pbt: %d workers %3.1fs' % (len(workers), pbt_time.elapsed()))


def dup_hparams(dest, source):
    dest['hparams'] = copy.copy(source['hparams'])


def dup_weights(dest, source):
    dest['dup_from_name'] = source['name']


def resample_hparams(worker):
    worker['hparams'] = [fun() for fun in worker['resample']]


def perturb_hparams(worker):
    worker['hparams'] = [fun(param) for fun, param in zip(
        worker['perturb'], worker['hparams'])]


def truncate_pop(workers, cutoff=0.2, dup_all=True, explore_fun=None):
    ranked = sorted(workers, key=lambda worker: worker['score'], reverse=True)
    index = int(cutoff * len(ranked))
    for best, worst in zip(ranked[:index], ranked[-index:]):
        dup_weights(worst, best)
        if dup_all:
            dup_hparams(worst, best)
        if explore_fun:
            explore_fun(worst)


def perturb(hparam, min_=0.0, max_=1.0, scale=[0.9, 1.1]):
    return np.clip(hparam * randbeta(*scale), min_, max_)


def randbeta(min_=0, max_=1, a=0.2, b=0.2):
    return min_ + (max_ - min_) * np.random.beta(a, b)


def main():
    dataset = 'mock'
    hparams_fun = [np.random.random]
    perturb_fun = [perturb]
    popsize = 10

    workers = []
    for i in range(popsize):
        name = 'ckpt/worker_' + str(i) + '.ckpt'
        hparams = [fun() for fun in hparams_fun]
        worker = {'name': name, 'dup_from_name': None, 'id': i, 'score': np.random.random(),
                  'hparams': hparams, 'resample': hparams_fun, 'perturb': perturb_fun,
                  'dataset': dataset}
        workers.append(worker)

    for step in range(1, 2 + 1):
        print('step %d' % step)
        for worker in workers:
            for key in ['name', 'dup_from_name', 'id', 'score', 'hparams', 'dataset']:
                print(key, worker[key], end=', ')
            print('')
        pbt(workers, 0.5)


if __name__ == '__main__':
    main()
