from __future__ import print_function

import numpy as np
import copy


def pbt(workers):
    print('pbt:', len(workers), 'workers')
    dup_hparams(workers[1], workers[0])
    dup_weights(workers[1], workers[0])
    resample_hparams(workers[0])


def dup_hparams(dest, source):
    dest['hparams'] = copy.copy(source['hparams'])


def dup_weights(dest, source):
    dest['dup_from_name'] = source['name']


def resample_hparams(worker):
    worker['hparams'] = [fun() for fun in worker['resample']]
