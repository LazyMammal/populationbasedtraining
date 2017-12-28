from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt

from pbt import Worker, PBT


def surrogate_figure2(t, h):
    # 1.2 - (h1 * t1**2 + h2 * t2**2)
    a = tf.constant(1.2)
    p = tf.constant(2.0)
    return a - tf.reduce_sum(tf.multiply(h, tf.pow(t, p)))


def objective_figure2(t):
    # 1.2 - (t1**2 + t2**2)
    return 1.2 - (t[0]**2 + t[1]**2)


def eval_figure2(worker):
    return objective_figure2(worker.nn)


def train_figure2(worker, steps=1):
    sess = tf.get_default_session()
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        theta = tf.get_variable("weights", initializer=tf.constant(worker.nn))
        h = tf.placeholder_with_default(tf.constant([1.0, 1.0]), shape=[2])
        yofx = surrogate_figure2(theta, h)
        cost = -yofx

        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        sess.run(init)
        for _ in range(steps):
            sess.run(optimizer, feed_dict={h: worker.hyperparams})
        worker.nn = theta.eval()


def nulltrain(worker):
    # dotrain( worker.nn, worker.hyperparams )
    pass


def randeval(worker):
    # return doeval( worker.nn )
    if worker.score > 0.0:
        return worker.score
    else:
        return np.random.random()


def make_plot(exploit, explore, poplist):
    make_plot.num += 1

    if exploit and explore:
        title = "PBT"
    elif exploit:
        title = "exploit"
    elif explore:
        title = "explore"
    else:
        title = "only train"

    plt.subplot(make_plot.num)
    plt.title(title)
    for m in range(len(poplist[0])):
        plt.plot([w[m][0] for w in poplist])

    plt.subplot(make_plot.num + 4)
    for m, c, marker in zip(range(len(poplist[0])), ['b', 'g', 'r'], ['^', 'v', 'o']):
        plt.plot([w[m][1] for w in poplist], [w[m][2]
                                              for w in poplist], c=c, marker=marker)


make_plot.num = 240


def main():
    """ make a plot from figure 2 in the PBT paper """
    # np.random.seed(0)
    with tf.Session() as sess:

        for exploit in [True, False]:
            for explore in [True, False]:

                population = PBT(pop=[Worker([0.0, 1.0], [0.9, 0.9]),
                                      Worker([1.0, 0.0], [0.9, 0.9])])
                population.testpop(eval_figure2)
                poplist = [[[w.score, w.nn[0], w.nn[1]]
                            for w in population.pop]]

                for step in range(10):
                    for worker in population.pop:
                        train_figure2(worker, 4)

                    population.testpop(eval_figure2)

                    if exploit:
                        population.exploit(0.5)

                    if explore:
                        population.explore(0.5)

                    poplist.append([[w.score, w.nn[0], w.nn[1]]
                                    for w in population.pop])

                make_plot(exploit, explore, poplist)
    plt.show()


main()
