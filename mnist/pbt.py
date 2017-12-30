from __future__ import print_function

import numpy as np
import copy


def pbt(workers):
    print('pbt:', len(workers), 'workers')
    truncate_pop(workers, explore_fun=perturb_hparams)


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
