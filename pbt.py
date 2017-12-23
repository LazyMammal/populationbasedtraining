from __future__ import print_function

import numpy as np

'''
operations:
* exploit_truncate (cutoff=20%)
* explore_perturb (elite=20%)

config:
* start popsize
* list of operations
* list of parameters to tune (ranges, distributions)
* training function
* testing function

loop:
* train
* test
* rank
* winnow
* spawn
'''

# np.random.seed(0)

class Worker:
    def __init__(self, hyperparams=None, nn=None):
        self.score = 0.0
        self.hyperparams = hyperparams
        self.nn = nn

    def __repr__(self):
        return repr((id(self), self.score, self.hyperparams)) # , self.nn

    def dup(self, worker):
        self.score = worker.score
        self.hyperparams = worker.hyperparams
        self.nn = worker.nn

def nulltrain(worker):
    # dotrain( worker.nn, worker.hyperparams )
    pass

def randeval(worker):
    # return doeval( worker.nn )
    if worker.score > 0.0:
        return worker.score
    else:
        return np.random.random()

def trainpop(pop, train):
    for worker in pop:
        train(worker)

def testpop(pop, test):
    for worker in pop:
        worker.score = test(worker)

def truncate(pop, cutoff=0.2):
    pop.sort(key=lambda worker: worker.score, reverse=True)
    for best,worst in zip(pop[:int(cutoff*len(pop))], pop[-int(cutoff*len(pop)):]):
        worst.dup(best)

population = [Worker() for _ in range(20)]
print(population)

for step in range(3):
    trainpop(population, nulltrain)
    testpop(population, randeval)
    truncate(population)

    print("Step:", step)
    print(population)
