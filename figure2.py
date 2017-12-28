from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy
import timeit
import matplotlib.pyplot as plt

from pbt import Worker, PBT


def surrogate_figure2(t, h):
    # 1.2 - (h1 * t1**2 + h2 * t2**2)
    a = tf.constant(1.2, dtype=tf.float64)
    p = tf.constant(2.0, dtype=tf.float64)
    return a - tf.reduce_sum(tf.multiply(h, tf.pow(t, p)))


def objective_figure2(t):
    # 1.2 - (t1**2 + t2**2)
    return 1.2 - (t[0]**2 + t[1]**2)


def eval_figure2(worker):
    return objective_figure2(worker.nn)


def make_tf_model(theta_init=[0.9, 0.9], h_init=[1.0, 1.0]):
    sess = tf.get_default_session()
    # with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    theta = tf.get_variable("theta", initializer=tf.constant(
        theta_init, dtype=tf.float64), dtype=tf.float64)
    theta_update_placeholder = tf.placeholder(theta.dtype, shape=theta.shape)
    theta_update = theta.assign(theta_update_placeholder).op

    h = tf.placeholder_with_default(tf.constant(
        h_init, dtype=tf.float64), shape=[2])
    yofx = surrogate_figure2(theta, h)
    cost = -yofx

    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    sess.run(init)

    return theta, h, optimizer, theta_update, theta_update_placeholder


def train_figure2(worker, steps, theta, h, optimizer, theta_update, theta_update_placeholder):
    sess = tf.get_default_session()
    sess.run(theta_update, feed_dict={theta_update_placeholder: worker.nn})
    for _ in range(steps):
        sess.run(optimizer, feed_dict={h: worker.hyperparams})
    worker.nn = theta.eval()


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
    plt.ylim(-0.5, 1.25)
    for m, c, marker in zip(range(len(poplist[0])), ['b', 'g', 'r'], ['^', 'v', 'o']):
        plt.plot([w[m][0] for w in poplist], c=c, alpha=0.5)

    ax = plt.subplot(make_plot.num + 4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for m, c, marker in zip(range(len(poplist[0])), ['b', 'g', 'r'], ['^', 'v', 'o']):
        plt.scatter([w[m][1] for w in poplist],
                    [w[m][2] for w in poplist], c=c, marker=marker, alpha=0.5)


make_plot.num = 240


def main():
    """ make a plot from figure 2 in the PBT paper """
    # np.random.seed(0)
    with tf.Session() as sess:

        theta, h, optimizer, theta_update, theta_update_placeholder = make_tf_model()

        start_time = timeit.default_timer()
        split_start = start_time
        for exploit in [True, False]:
            for explore in [True, False]:

                population = PBT(pop=[Worker([0.0, 1.0], [0.9, 0.9], perturbscale=[0.9, 1.1], jitter=0.1),
                                      Worker([1.0, 0.0], [0.9, 0.9], perturbscale=[0.9, 1.1], jitter=0.1)])
                population.testpop(eval_figure2)
                poplist = [[[w.score, w.nn[0], w.nn[1]]
                            for w in population.pop]]

                for step in range(20):

                    for _ in range(4):
                        for worker in population.pop:
                            train_figure2(worker, 4, theta, h, optimizer,
                                          theta_update, theta_update_placeholder)
                        poplist.append([[eval_figure2(w), w.nn[0], w.nn[1]]
                                        for w in population.pop])

                    population.testpop(eval_figure2)

                    if exploit:
                        population.exploit(0.5)

                    if explore:
                        population.explore(0.5)

                    for idx, worker in enumerate(population.pop):
                        print(idx, worker)
                    split_end = timeit.default_timer()
                    print((split_end - split_start), "/", (split_end - start_time), "s")
                    split_start = split_end
                print('-----')

                make_plot(exploit, explore, poplist)
    plt.show()


main()
