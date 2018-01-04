from __future__ import print_function

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def main(args):
    table = np.loadtxt(args.file, delimiter=',', usecols=(0, 1, 5, 6, 7, 8), skiprows=args.skiprows)
    maxsteps  = int(max(np.array(table)[:, 0]))
    maxworker = int(max(np.array(table)[:, 1]) + 1)
    workerorder = np.reshape(table[np.lexsort((table[:,0],table[:,1]))], (maxworker, -1, 6))
    steporder = np.reshape(table[np.lexsort((table[:,1],table[:,0]))], (maxsteps, -1, 6))

    if args.logplot:
        mainplots(workerorder, [None, None, None, 2])
    else:
        mainplots(workerorder)
    overfit(workerorder)
    gridplot(steporder)


def gridplot(steporder, gridshape=(7,7)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-135, elev=45)
    ax.set_xlabel('log(batch size)')
    ax.set_ylabel('log(learn rate)')
    ax.set_zlabel('test accuracy')
    for step in steporder:
        x = np.log(np.reshape(step[:, 3], gridshape))
        y = np.log(np.reshape(step[:, 2], gridshape))
        z = np.reshape(step[:, 5], gridshape)
        print(z)
        ax.plot_wireframe(x, y, z)
    plt.show()


def overfit(workerorder):
    plt.subplot(121)
    plt.title('% overfit')
    plt.xlabel("steps")
    plt.ylabel("(test - train) / test")
    for worker in workerorder:
        plt.plot(worker[:, 0], ((1-worker[:, 5]) - (1-worker[:, 4])) / (1-worker[:, 5]), alpha=0.5, marker='o')

    plt.subplot(122)
    plt.title('test / train')
    plt.xlabel("steps")
    plt.ylabel("test / train")
    for worker in workerorder:
        plt.plot(worker[:, 0], (1-worker[:, 5]) / (1-worker[:, 4]), alpha=0.5, marker='o')
    plt.show()


def mainplots(workerorder, logargs=[None, None, None, None], yscaleargs=[(0.0,1.0),(0.0,1.0),None,None]):
    for col, title, plotnum, logplot, yscale in [(a, b, c, d, e) for (a, b, c), d, e in zip([(5, 'test accuracy', 221), (4, 'train accuracy', 222), (3, 'batch size', 223), (2, 'learning rate', 224)], logargs, yscaleargs)]:
        plt.subplot(plotnum)
        if not logplot is None:
            plt.semilogy(logplot)
        if not yscale is None:
            plt.ylim(*yscale)
        plt.title(title)
        for worker in workerorder:
            plt.plot(worker[:, 0], worker[:, col], alpha=0.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='?',
                        default="csv\workers.csv", help="input file")
    parser.add_argument('--skiprows', nargs='?', type=int,
                        default=0, help="Skip the first skiprows lines; default: 0")
    parser.add_argument('--logplot', nargs='?', type=bool,
                        default=False, help="Use log scale y-axis for hyperparameters")
    main(parser.parse_args())
