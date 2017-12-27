# population based training

some experiments in PBT

reference: [Population Based Training of Neural Networks, Jaderberg et al](https://arxiv.org/abs/1711.09846)

## roadmap

operations:

* shrink_population (pctshrink=5%, fixedshrink=5, minsize=10)
* exploit_truncate (cutoff=20%)
* explore_resample | explore_perturb (elite=20%) | explore_recombine (elite=20%)
* net_deeper | net_wider | net_shrink | net_perturb | net_resample | net_recombine

config:

* start popsize
* list of operations (likelihoods, parameters)
* list of parameters to tune (ranges, distributions)
* training function
* testing function

experiments:

* contracting popsize
* recombination (hyperparams, weights)
* architecture (net2deeper, net2wider, sparse, randomize weight)
* meta-PBT (optimize PBT config using PBT)
