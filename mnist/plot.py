from __future__ import print_function

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    table = np.loadtxt(args.file, delimiter=',', usecols=(0, 1, 5, 6, 7, 8), skiprows=args.skiprows)
    maxsteps  = int(max(np.array(table)[:, 0]))
    maxworker = int(max(np.array(table)[:, 1]) + 1)
    steporder = np.reshape(table, (maxsteps, -1, 6))
    workerorder = np.swapaxes(steporder, 0, 1)

    print(workerorder)

    mainplots(workerorder)
    overfit(workerorder)

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

def mainplots(workerorder):
    for col, title, plotnum in [(5, 'test accuracy', 221), (4, 'train accuracy', 222), (3, 'batch size', 223), (2, 'learning rate', 224)]:
        plt.subplot(plotnum)
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
    main(parser.parse_args())
