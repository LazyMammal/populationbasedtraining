from __future__ import print_function

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    table = np.loadtxt(args.file, delimiter=',', usecols=(0, 1, 5, 6, 7, 8))
    steporder = np.reshape(table, (10, -1, 6))
    workerorder = np.swapaxes(steporder, 0, 1)
    for worker in workerorder:
        plt.plot(worker[:, 0], worker[:, 5], alpha=0.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='?',
                        default="csv\workers.csv", help="input file")
    main(parser.parse_args())
