from __future__ import print_function

import numpy as np
import pbt


def resample_learnrate():
    return 2.0**(-3 - 15 * np.random.random())


def resample_batchsize():
    return int(np.random.logseries(.95) * 10)


def perturb_learnrate(learnrate):
    return pbt.perturb(learnrate, 0.0, 0.1)


def perturb_batchsize(batchsize):
    return int(pbt.perturb(batchsize / 10.0, 1, 100) * 10)
