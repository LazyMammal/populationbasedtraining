from __future__ import print_function

import copy
import numpy as np


class Worker:
    def __init__(self, hyperparams=[1.0], nn=[1.0], explore=None, perturbscale=[0.5, 2.0], jitter=0.1, cliprange=(None, None)):
        self.score = 0.0
        self.hyperparams = np.array(hyperparams)
        self.nn = np.array(nn)
        self.func_explore = explore or Worker.perturbbeta
        self.perturbscale = perturbscale
        self.jitter = jitter
        self.cliprange = cliprange

    def __repr__(self):
        return repr((id(self), self.score, self.hyperparams, self.nn))

    def dup(self, worker):
        self.score = worker.score
        self.hyperparams = copy.copy(worker.hyperparams)
        self.nn = copy.copy(worker.nn)

    def dupweights(self, worker):
        self.nn = copy.copy(worker.nn)

    def explore(self):
        self.func_explore(self)

    def perturbbeta(self):
        self.hyperparams[:] = np.array(
            [param * randbeta(self.perturbscale[0], self.perturbscale[1]) +
             self.jitter * (np.random.random() - 0.5) for param in self.hyperparams])
        self.clip()

    def perturb(self):
        self.hyperparams[:] = np.array(
            [param * np.random.choice(self.perturbscale) +
             self.jitter * (np.random.random() - 0.5) for param in self.hyperparams])
        self.clip()

    def resample(self):
        self.hyperparams[:] = np.array(
            [np.random.random() for param in self.hyperparams])
        if self.cliprange and self.cliprange != (None, None):
            min_, max_ = self.cliprange
            self.hyperparams = self.hyperparams * (max_ - min_) + min_

    def clip(self):
        if self.cliprange and self.cliprange != (None, None):
            min_, max_ = self.cliprange
            np.clip(self.hyperparams, min_, max_, out=self.hyperparams)


class PBT:
    def __init__(self, popsize=20, train=None, test=None, explore=None, pop=None, cliprange=None):
        if pop is None:
            self.pop = [Worker(explore=explore, cliprange=cliprange)
                        for _ in range(popsize)]
        else:
            self.pop = pop
        self.train = train
        self.test = test
        self.exploit = self.truncate

    def trainpop(self, train=None):
        if train is None:
            train = self.train
        if not train is None:
            for worker in self.pop:
                train(worker)

    def testpop(self, test=None):
        if test is None:
            test = self.test
        if not test is None:
            for worker in self.pop:
                worker.score = test(worker)

    def truncate(self, cutoff=0.2):
        ranked = sorted(
            self.pop, key=lambda worker: worker.score, reverse=True)
        index = int(cutoff * len(ranked))
        for best, worst in zip(ranked[:index], ranked[-index:]):
            worst.dupweights(best)

    def explore(self, cutoff=0.2):
        ranked = sorted(
            self.pop, key=lambda worker: worker.score, reverse=True)
        index = int(cutoff * len(ranked))
        for worst in ranked[-index:]:
            worst.explore()


def randbeta(min_=0, max_=1, a=0.2, b=0.2):
    return min_ + (max_ - min_) * np.random.beta(a, b)
