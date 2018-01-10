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
        0, 1, 5, 6, 7, 8, 9), skiprows=args.skiprows)
    # maxsteps  = int(max(np.array(table)[:, 0]))
    # steporder = np.reshape(table[np.lexsort((table[:,1],table[:,0]))], (maxsteps, -1, 6))
    # gridplot(steporder)
    workerorder = group_by_worker(table)
    if args.outdir is '':
        outpath = None
    else:
        outpath = args.outdir + '/' + os.path.basename(args.file).split('.')[0]

    fig = plt.figure()
    plotcompare(workerorder, yaxis={'col': 2, 'label': 'learning rate', 'scale': 'log' if args.logplot else None})
    output_plot(outpath, '_lr')

    fig = plt.figure()
    plotcompare(workerorder, yaxis={'col': 5, 'label': 'test accuracy', 'limit': (0.0, 1.0)})
    output_plot(outpath, '_test')

    fig = plt.figure()
    plotcompare(workerorder, yaxis={'col': 5, 'label': 'test accuracy', 'limit': (0.0, 1.0)}, plotnum=221)
    plotcompare(workerorder, yaxis={'col': 4, 'label': 'train accuracy', 'limit': (0.0, 1.0)}, plotnum=222)
    plotcompare(workerorder, yaxis={'col': 6, 'label': 'validation accuracy', 'limit': (0.0, 1.0)}, plotnum=223)
    plotcompare(workerorder, yaxis={'col': 2, 'label': 'learning rate',
                                    'scale': 'log' if args.logplot else None}, plotnum=224)
    # plotcompare(workerorder, yaxis={'col': 3, 'label': 'batch size'}, plotnum=223)
    output_plot(outpath)

    overfit(workerorder, outpath)
    PQ(workerorder, outpath)

    fig = plt.figure()
    # plotcompare(workerorder, {'col': 3, 'label': 'batch size', 'scale': 'log' if args.logplot else None}, plotnum=121)
    plotcompare(workerorder, {'col': 2, 'label': 'learning rate',
                              'scale': 'log' if args.logplot else None, 'reverse': True})  # , plotnum=122)
    output_plot(outpath, '_params')


def output_plot(outpath, suffix=''):
    adjust_plots()
    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath + suffix + '.png')


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


def plotcompare(
        workerorder, xaxis={'col': 0, 'label': 'epoch'},
        yaxis={'col': 5, 'label': 'test accuracy', 'limit': (0.0, 1.0)},
        plotnum=None):
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
    '''
    # for x, worker in enumerate(workerorder):
    # for label, worker in zip(['GradDescent_1', 'GradDescent_2', 'Momentum_1', 'Momentum_2', 'RMSprop_1', 'RMSprop_2', 'Adam_1', 'Adam_2'], workerorder):
    # for label, worker in zip(['warm restarts', 'online PBT', 'grad descent', 'adam'], workerorder):
    # for label, worker in zip(['x1.5','x1.7','x2.0','x2.4','x3.0'], workerorder):
    for label, worker in zip(['x2, 1','x1, 11','x1, 44'], workerorder):
        a, = plt.plot(worker[:, xaxis['col']], worker[:, yaxis['col']], alpha=0.5) #, marker='o')
        # a.set_label(str(float(x+5)/10))
        a.set_label(label)
    plt.legend()
    '''
    for worker in workerorder:
        plt.plot(worker[:, xaxis['col']], worker[:, yaxis['col']], alpha=0.5) #, marker='o')


def gridplot(steporder, gridshape=(7, 7), outpath=None):
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
    output_plot(outpath, '_grid')


def overfit(workerorder, outpath=None):
    fig = plt.figure()
    plt.subplot(221)
    plt.title('% overfit')
    plt.xlabel("epoch")
    plt.ylabel("(test - train) / test")
    plt.ylim(0.0, 1.0)
    for worker in workerorder:
        plt.plot(worker[:, 0], overfit_score.overfit_accuracy(worker[:, 4], worker[:, 5]), alpha=0.5) #, marker='o')

    plt.subplot(223)
    plt.title('overfit blended (bigger is better)')
    plt.xlabel("epoch")
    plt.ylabel("(1.0 - overfit) * train_accuracy")
    plt.ylim(0.0, 1.0)
    for worker in workerorder:
        plt.plot(worker[:, 0], overfit_score.overfit_blended(worker[:, 4], worker[:, 5]), alpha=0.5) #, marker='o')

    plt.subplot(222)
    plt.title('test / train')
    plt.xlabel("epoch")
    plt.ylabel("test / train")
    for worker in workerorder:
        plt.plot(worker[:, 0], (1 - worker[:, 5]) / (1 - worker[:, 4]), alpha=0.5) #, marker='o')

    output_plot(outpath, '_overfit')


def PQ(workerorder, outpath=None):
    fig = plt.figure()
    plt.subplot(223)
    plt.title('GL - General Loss (filtered)')
    plt.xlabel("epoch")
    plt.ylabel("avg(test) / best(test)")
    for worker in workerorder:
        plt.plot(worker[:, 0], pq_score.gl_accuracy(worker[:, 10], worker[:, 8]), alpha=0.5) #, marker='o')

    plt.subplot(224)
    plt.title('P - Progress (filtered)')
    plt.xlabel("epoch")
    plt.ylabel("avg(train) / best(train)")
    for worker in workerorder:
        plt.plot(worker[:, 0], pq_score.p_accuracy(worker[:, 9], worker[:, 7]), alpha=0.5) #, marker='o')

    plt.subplot(221)
    plt.title('PQ - Generality to Progress Ratio')
    plt.xlabel("epoch")
    plt.ylabel("GL / P")
    for worker in workerorder:
        plt.plot(
            worker[:, 0],
            pq_score.pq_accuracy(worker[:, 9],
                                 worker[:, 7],
                                 worker[:, 10],
                                 worker[:, 8]),
            alpha=0.5) #, marker='o')
    output_plot(outpath, '_PQ')


def adjust_plots(left=.11, bottom=.1, right=.97, top=.94, wspace=.33, hspace=.45):
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='?', default="csv/workers.csv", help="input file")
    parser.add_argument('--outdir', nargs='?', default="", help="output directory (or interactive if not given)")
    parser.add_argument('--skiprows', nargs='?', type=int, default=0, help="Skip the first skiprows lines; default: 0")
    parser.add_argument('--logplot', nargs='?', type=bool, default=False,
                        help="Use log scale y-axis for hyperparameters")
    main(parser.parse_args())
