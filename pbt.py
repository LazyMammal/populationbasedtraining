from __future__ import print_function

import copy
import numpy as np


class Worker:
    def __init__(self, hyperparams=None, nn=None, explore=None, perturbscale=[0.5, 2.0], jitter=0.1):
        self.score = 0.0
        self.hyperparams = hyperparams or [1.0]
        self.nn = nn or [1.0]
        if explore is 'resample':
            self.explore = self.resample
        else:
            self.explore = self.perturbbeta
        self.perturbscale = perturbscale
        self.jitter = jitter

    def __repr__(self):
        return repr((id(self), self.score, self.hyperparams, self.nn))

    def dup(self, worker):
        self.score = worker.score
        self.hyperparams = copy.copy(worker.hyperparams)
        self.nn = copy.copy(worker.nn)

    def dupweights(self, worker):
        self.nn = copy.copy(worker.nn)

    def perturbbeta(self, perturbscale=None):
        if perturbscale is None:
            perturbscale = self.perturbscale
        self.hyperparams[:] = [
            param * randbeta(perturbscale[0], perturbscale[1]) + self.jitter * (np.random.random() - 0.5) for param in self.hyperparams]

    def perturb(self, perturbscale=None):
        if perturbscale is None:
            perturbscale = self.perturbscale
        self.hyperparams[:] = [
            param * np.random.choice(perturbscale) + self.jitter * (np.random.random() - 0.5) for param in self.hyperparams]

    def resample(self):
        if not self.hyperparams is None:
            pass


class PBT:
    def __init__(self, popsize=20, train=None, test=None, explore=None, pop=None):
        if pop is None:
            self.pop = [Worker(explore=explore) for _ in range(popsize)]
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
        ranked = sorted(self.pop, key=lambda worker: worker.score, reverse=True)
        index = int(cutoff * len(ranked))
        for best, worst in zip(ranked[:index], ranked[-index:]):
            worst.dupweights(best)

    def explore(self, cutoff=0.2):
        ranked = sorted(self.pop, key=lambda worker: worker.score, reverse=True)
        index = int(cutoff * len(ranked))
        for worst in ranked[-index:]:
            worst.explore()


def randbeta(min_=0, max_=1, a=0.2, b=0.2):
    return min_ + (max_ - min_) * np.random.beta(a, b)
