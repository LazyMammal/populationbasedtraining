from __future__ import print_function

import numpy as np
import pbt


def resample_learnrate():
    return 2.0**(-3 - 15 * np.random.random())


def resample_batchsize():
    return int(np.clip(np.random.logseries(.995), 1, 1000))


def perturb_learnrate(learnrate):
    return pbt.perturb(learnrate, 0.0, 0.1, scale=[0.4, 2.1])


def perturb_batchsize(batchsize):
    return int(pbt.perturb(batchsize, 0.5, 1000, scale=[0.4, 2.1])+0.5)
