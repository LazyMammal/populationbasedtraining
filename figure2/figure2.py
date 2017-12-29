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
    return 1.2 - np.sum(np.power(t, 2))


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


def make_plot(title, plotnum, poplist):
    ax = plt.subplot(plotnum + 2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Q(θ)")
    plt.ylim(-0.5, 1.2)
    plt.xticks(np.arange(7 + 1))
    for m, c, marker in zip(range(len(poplist[0])), ['b', 'g', 'r'], ['^', 'v', 'o']):
        plt.plot([w[m][3] for w in poplist],
                 [w[m][0] for w in poplist], c=c, alpha=0.5)

    ax = plt.subplot(plotnum)
    plt.title(title)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for m, c, marker in zip(range(len(poplist[0])), ['b', 'g', 'r'], ['^', 'v', 'o']):
        plt.scatter([w[m][1] for w in poplist],
                    [w[m][2] for w in poplist], c=c, marker=marker, alpha=0.5)


def main():
    """ make a plot from figure 2 in the PBT paper """
    # np.random.seed(0)
    plt.subplots_adjust(left=.085, bottom=.1, right=.975,
                        top=.95, wspace=.45, hspace=.45)

    with tf.Session() as sess:

        theta, h, optimizer, theta_update, theta_update_placeholder = make_tf_model()

        start_time = timeit.default_timer()
        split_start = start_time
        for exploit, explore, name, plotnum in [
            (True, True, "PBT", 241),
            (False, True, "Explore only", 242),
            (True, False, "Exploit only", 245),
            (False, False, "Grid search", 246)
        ]:
            population = PBT(pop=[Worker([0.0, 1.0], [0.9, 0.9], perturbscale=[0.5, 2.0], jitter=0.5, cliprange=(0, 100)),
                                  Worker([1.0, 0.0], [0.9, 0.9], perturbscale=[0.5, 2.0], jitter=0.5, cliprange=(0, 100))])
            population.testpop(eval_figure2)
            poplist = [[[w.score, w.nn[0], w.nn[1], 0]
                        for w in population.pop]]

            for step in range(0, 7):

                train_steps = 5
                for substep in range(train_steps):
                    for worker in population.pop:
                        train_figure2(worker, 4, theta, h, optimizer,
                                      theta_update, theta_update_placeholder)
                    poplist.append([[eval_figure2(w), w.nn[0], w.nn[1], (step + substep / float(train_steps))]
                                    for w in population.pop])

                population.testpop(eval_figure2)

                if exploit:
                    population.exploit(0.5)

                if explore:
                    population.explore(0.5)

                for idx, worker in enumerate(population.pop):
                    print(idx, worker)
                split_end = timeit.default_timer()
                print((split_end - split_start), "/",
                      (split_end - start_time), "s")
                split_start = split_end
            print('-----')

            make_plot(name, plotnum, poplist)
    plt.show()


main()
