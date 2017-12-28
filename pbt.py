from __future__ import print_function

import numpy as np


class Worker:
    def __init__(self, hyperparams=None, nn=None, explore=None, perturbscale=[0.8,1.2]):
        self.score = 0.0
        self.hyperparams = hyperparams or [1.0]
        self.nn = nn
        if explore is 'resample':
            self.explore = self.resample
        else:
            self.explore = self.perturb
        self.perturbscale = perturbscale

    def __repr__(self):
        return repr((id(self), self.score, self.hyperparams))  # , self.nn

    def dup(self, worker):
        self.score = worker.score
        self.hyperparams = worker.hyperparams
        self.nn = worker.nn

    def perturb(self, perturbscale=None):
        if perturbscale is None:
            perturbscale = self.perturbscale
        self.hyperparams[:] = [param * np.random.choice(self.perturbscale) for param in self.hyperparams]

    def resample(self):
        if not self.hyperparams is None:
            pass


class PBT:
    def __init__(self, popsize=20, train=None, test=None, explore=None):
        self.pop = [Worker(explore=explore) for _ in range(popsize)]
        self.train = train
        self.test = test

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
        self.pop.sort(key=lambda worker: worker.score, reverse=True)
        index = int(cutoff * len(self.pop))
        for best, worst in zip(self.pop[:index], self.pop[-index:]):
            worst.dup(best)
            worst.explore()
