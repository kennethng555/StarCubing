# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:27:16 2019

@author: Kenneth Ng
@email: kenng7183@gmail.com
@github: kennethng555
"""

import os
import psutil
import time
from StarCube import StarCube
from Tree import Tree
from Node import Node

import pandas as pd
import numpy as np
import math
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import sys


# input: data table
# output: filtered data table with unique entry counts
def gen_startable(df):
    replaced = 0
    if df.keys()[0] == 0:
        df = df.rename(columns={0: 'A'})
        replaced = 1

    unique = []
    counts = []
    for i in range(len(df.columns)):
        unique = np.unique(df.iloc[:, i])
        counts.append([])
        for j in range(len(unique)):
            count = len(df.iloc[:, i][df.iloc[:, i] == unique[j]])
            counts[i].append(count)
            if threshold > count:
                df.iloc[:, i][df.iloc[:, i] == unique[j]] = -1

    df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

    if replaced == 1:
        df = df.rename(columns={'A': 0})
    return df


def discretize(df, NumericalDims, NoPartitions):
    ### Equilength Binning
    Xmax = [0] * len(df.columns)
    Xmin = [0] * len(df.columns)
    dX = [0] * len(df.columns)
    # Scan 1:
    # Determine the max and min of each column
    for j in NumericalDims:
        ColMax = df.iloc[0, j];
        ColMin = df.iloc[0, j];
        for i in range(len(df)):
            if df.iloc[i, j] < ColMin:
                ColMin = df.iloc[i, j]
            if df.iloc[i, j] > ColMax:
                ColMax = df.iloc[i, j]
        Xmax[j] = ColMax
        Xmin[j] = ColMin
        dX[j] = (Xmax[j] - Xmin[j]) / (NoPartitions - 1)
    # Scan 2:
    # Replace numerical data with it's discretized form (represented as an integer)
    for j in NumericalDims:
        for i in range(len(df)):
            # In this line, you reassign each double to an integer based on it's bin
            df.iloc[i, j] = math.floor((df.iloc[i, j] - Xmin[j]) / dX[j])
    return df.astype(int)

# turns integer categories into strings with labels
def decode(startable):
    separator = '_'
    separator2 = ''
    for i in range(len(startable)):
        for j in range(len(startable.columns) - 1):
            startable.iloc[i, j] = separator.join(
                [separator2.join([chr(97 + math.floor(j / 26)) * (math.floor(j / 26)), chr(97 + (j % 26))]),
                 str(int(startable.iloc[i, j]))])
    return startable


if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    start_time = time.time()
    f = open("../output.txt", "w+")

    ## uncomment to export events to a text file
    # original = sys.stdout
    # sys.stdout = open('events.txt', 'w')

    file = "data/Test.csv"
    print('input file: ', file)
    df = pd.read_csv(file)
    threshold = 2
    # NumericalDims = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    # NoPartitions = 2

    print('Attempting to discretize the imported dataset...')
    # df = discretize(df, NumericalDims, NoPartitions)
    print('Proceeding to star-cubing algorithm...')

    startable = gen_startable(df)
    startable = decode(startable)

    root = Node(data='root', count=sum(startable.iloc[:, len(startable.columns) - 1]), depth=0)
    startree = Tree(root, threshold)
    startree.generate(startable)

    starcube = StarCube(threshold, startree)
    starcube.starcubing(startree, root)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Total Bytes Used: ", end='')
    print(process.memory_info().rss)

    f.close()

    # f= open("output.txt","r")
    # if f.mode == 'r':
    #    contents =f.readlines()

    # for i in range(len(contents)):
    #    contents[i] = contents[i].rstrip("\n\r").split(',')
    # contents[0][len(contents[0])-1]

    # fig = pyplot.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(df["class"],df["lymphatics"],df["block of affere"], c=df["class"])
    # pyplot.show()
    # del fig

    g = sns.PairGrid(df)
    g.map_diag(sns.stripplot)
    g.map_offdiag(sns.stripplot, jitter=True)

    # sys.stdout = original
