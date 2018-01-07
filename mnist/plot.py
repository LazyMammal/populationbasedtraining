from __future__ import print_function

import sys
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import overfit_score
import pq_score


def main(args):
    table = np.loadtxt(args.file, delimiter=',', usecols=(
        0, 1, 5, 6, 7, 8), skiprows=args.skiprows)
    # maxsteps  = int(max(np.array(table)[:, 0]))
    # steporder = np.reshape(table[np.lexsort((table[:,1],table[:,0]))], (maxsteps, -1, 6))
    # gridplot(steporder)
    workerorder = group_by_worker(table)
    outpath = args.outdir + os.path.basename(args.file).split('.')[0]

    fig = plt.figure()
    plotcompare(workerorder, yaxis={
                'col': 5, 'label': 'test accuracy', 'limit': (0.0, 1.0)}, plotnum=221)
    plotcompare(workerorder, yaxis={
                'col': 4, 'label': 'train accuracy', 'limit': (0.0, 1.0)}, plotnum=222)
    plotcompare(workerorder, yaxis={
                'col': 3, 'label': 'batch size'}, plotnum=223)
    plotcompare(workerorder, yaxis={'col': 2, 'label': 'learning rate',
                                    'scale': 'log' if args.logplot else None}, plotnum=224)
    adjust_plots()
    # plt.show()
    plt.savefig(outpath + '.png')

    overfit(workerorder, outpath)
    PQ(workerorder, outpath)

    fig = plt.figure()
    plotcompare(workerorder, {'col': 3, 'label': 'batch size',
                              'scale': 'log' if args.logplot else None}, plotnum=121)
    plotcompare(workerorder, {'col': 2, 'label': 'learning rate',
                              'scale': 'log' if args.logplot else None, 'reverse': True}, plotnum=122)
    adjust_plots()
    # plt.show()
    plt.savefig(outpath + '_params.png')


def group_by_worker(table, bestcol=(4, 5), avgcol=(4, 5), decay=0.5):
    workerorder = []
    workerblock = []
    best = []
    avg = np.array([])
    wid = None
    for row in table[np.lexsort((table[:, 0], table[:, 1]))]:
        if workerblock and wid != row[1]:
            workerorder.append(np.array(workerblock))
            workerblock = []
            best = []
            avg = np.array([])
        if not len(best):
            best = [row[col] for col in bestcol]
        if not len(avg):
            avg = [row[col] for col in avgcol]
        wid = row[1]
        best = np.fmax(best, [row[col] for col in bestcol])
        avg = np.array(avg) * (1.0 - decay)
        avg += np.array([row[col] for col in avgcol]) * decay
        workerblock.append(np.append(np.array(row), [best, avg]))
    workerorder.append(np.array(workerblock))
    return workerorder


def plotcompare(workerorder, xaxis={'col': 0, 'label': 'step'}, yaxis={'col': 5, 'label': 'test accuracy', 'limit': (0.0, 1.0)}, plotnum=None):
    if not plotnum is None:
        plt.subplot(plotnum)
    plt.xlabel(xaxis['label'])
    plt.ylabel(yaxis['label'])
    if 'limit' in xaxis:
        plt.xlim(*xaxis['limit'])
    if 'limit' in yaxis:
        plt.ylim(*yaxis['limit'])
    if 'scale' in xaxis and xaxis['scale']:
        plt.xscale(xaxis['scale'])
    if 'scale' in yaxis and yaxis['scale']:
        plt.yscale(yaxis['scale'])
    if 'reverse' in xaxis:
        plt.gca().invert_xaxis()
    if 'reverse' in yaxis:
        plt.gca().invert_yaxis()
    for worker in workerorder:
        plt.plot(worker[:, xaxis['col']],
                 worker[:, yaxis['col']], alpha=0.5, marker='o')


def gridplot(steporder, gridshape=(7, 7), outpath='plot'):
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
        ax.plot_wireframe(x, y, z, alpha=0.85)
    # plt.show()
    plt.savefig(outpath + '_grid.png')


def overfit(workerorder, outpath='plot'):
    fig = plt.figure()
    plt.subplot(221)
    plt.title('% overfit')
    plt.xlabel("steps")
    plt.ylabel("(test - train) / test")
    plt.ylim(0.0, 1.0)
    for worker in workerorder:
        plt.plot(worker[:, 0], overfit_score.overfit_accuracy(
            worker[:, 4], worker[:, 5]), alpha=0.5, marker='o')

    plt.subplot(223)
    plt.title('overfit blended (bigger is better)')
    plt.xlabel("steps")
    plt.ylabel("(1.0 - overfit) * test_accuracy")
    plt.ylim(0.0, 1.0)
    for worker in workerorder:
        plt.plot(worker[:, 0], overfit_score.overfit_blended(
            worker[:, 4], worker[:, 5]), alpha=0.5, marker='o')

    plt.subplot(222)
    plt.title('test / train')
    plt.xlabel("steps")
    plt.ylabel("test / train")
    for worker in workerorder:
        plt.plot(worker[:, 0], (1 - worker[:, 5]) /
                 (1 - worker[:, 4]), alpha=0.5, marker='o')

    plotcompare(workerorder, yaxis={
                'col': 6, 'label': 'validation accuracy', 'limit': (0.0, 1.0)}, plotnum=224)

    adjust_plots()
    # plt.show()
    plt.savefig(outpath + '_overfit.png')


def PQ(workerorder, outpath='plot'):
    fig = plt.figure()
    plt.subplot(223)
    plt.title('GL - General Loss (filtered)')
    plt.xlabel("steps")
    plt.ylabel("avg(test) / best(test)")
    for worker in workerorder:
        plt.plot(worker[:, 0], pq_score.gl_accuracy(
            worker[:, 9], worker[:, 7]), alpha=0.5, marker='o')

    plt.subplot(224)
    plt.title('P - Progress (filtered)')
    plt.xlabel("steps")
    plt.ylabel("avg(train) / best(train)")
    for worker in workerorder:
        plt.plot(worker[:, 0], pq_score.p_accuracy(
            worker[:, 8], worker[:, 6]), alpha=0.5, marker='o')

    plt.subplot(221)
    plt.title('PQ - Generality to Progress Ratio')
    plt.xlabel("steps")
    plt.ylabel("GL / P")
    for worker in workerorder:
        plt.plot(worker[:, 0], pq_score.pq_accuracy(worker[:, 8], worker[:, 6],
                                                    worker[:, 9], worker[:, 7]), alpha=0.5, marker='o')
    adjust_plots()
    # plt.show()
    plt.savefig(outpath + '_PQ.png')


def adjust_plots(left=.11, bottom=.1, right=.97, top=.94, wspace=.33, hspace=.45):
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='?',
                        default="csv/workers.csv", help="input file")
    parser.add_argument('--outdir', nargs='?',
                        default="charts/", help="output directory (./charts)")
    parser.add_argument('--skiprows', nargs='?', type=int,
                        default=0, help="Skip the first skiprows lines; default: 0")
    parser.add_argument('--logplot', nargs='?', type=bool,
                        default=False, help="Use log scale y-axis for hyperparameters")
    main(parser.parse_args())
