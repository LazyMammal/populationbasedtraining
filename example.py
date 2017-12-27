from __future__ import print_function

import numpy as np
from pbt import Worker, PBT

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


def nulltrain(worker):
    # dotrain( worker.nn, worker.hyperparams )
    pass


def randeval(worker):
    # return doeval( worker.nn )
    if worker.score > 0.0:
        return worker.score
    else:
        return np.random.random()


population = PBT()
print(population.pop)

for step in range(3):
    population.trainpop(nulltrain)
    population.testpop(randeval)
    population.truncate()

    print("Step:", step)
    print(population.pop)
