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

import pandas as pd
import numpy as np
import math
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import sys

process = psutil.Process(os.getpid())
start_time = time.time()
f = open("output.txt", "w+")


class Node(object):
    def __init__(self, data=None, count=0, left_node=None, right_node=None, parent=None, depth=None):
        self.data = data
        self.count = count
        self.left_node = left_node
        self.right_node = right_node
        self.parent = parent
        self.depth = depth

    def get_data(self):
        return self.data

    def get_count(self):
        return self.count

    def get_left(self):
        return self.left_node

    def get_right(self):
        return self.right_node

    def get_parent(self):
        return self.parent

    def set_count(self, new_count):
        self.count = new_count

    def set_left(self, new_left):
        self.left_node = new_left

    def set_right(self, new_right):
        self.right_node = new_right

    def set_parent(self, new_parent):
        self.parent = new_parent


class Tree(object):
    COUNT = [10]

    def __init__(self, root=None, threshold=None, neighbor=None, parent=None):
        self.root = root
        self.head = root
        self.threshold = threshold
        self.neighbor = neighbor
        self.parent = parent

    def insert_l(self, data, parent):
        new_node = Node(data=data, count=len(data), parent=self.head)
        self.head.set_left(new_node)
        self.head = new_node

    def insert_r(self, data, parent):
        new_node = Node(data=data, count=len(data), parent=self.head.get_parent)
        self.head.set_right(new_node)
        self.head = new_node

    def insert_left(self, node):
        self.head.set_left(node)
        self.head = node

    def insert_right(self, node):
        self.head.set_right(node)
        self.head = node

    # input: data(star) table
    # generates the star tree based on the datatable by inserting or aggregating each column as a new child
    def generate(self, df):
        num_dim = len(df.columns) - 1
        for i in range(len(df)):
            for j in range(num_dim - 1):
                if (self.head.get_left() == None):
                    new_left = Node(data=df.iloc[i, j], count=df.iloc[i, num_dim], parent=self.head, depth=j)
                    self.head.set_left(new_left)
                    self.head = self.head.get_left()
                else:
                    self.head = self.head.get_left()
                    while self.head.get_right() != None and self.head.get_data() != df.iloc[i, j]:
                        self.head = self.head.get_right()
                    if self.head.get_right() == None and self.head.get_data() != df.iloc[i, j]:
                        new_right = Node(data=df.iloc[i, j], count=df.iloc[i, num_dim], parent=self.head.get_parent(),
                                         depth=j)
                        self.head.set_right(new_right)
                        self.head = self.head.get_right()
                    elif self.head.get_data() == df.iloc[i, j]:
                        self.head.set_count(self.head.get_count() + df.iloc[i, num_dim])
            self.head = self.root


class StarCube(object):
    def __init__(self, min_sup=None, base_tree=None):
        self.min_sup = min_sup
        self.result = []
        self.base_tree = base_tree
        self.c_tree = base_tree
        self.reset = 0

    def insert_neighbor(self, tree):
        self.c_tree.neighbor = tree
        tree.parent = self.c_tree
        self.c_tree = self.c_tree.neighbor
        # self.c_tree = self.base_tree

    def delete_neighbor(self, tree):
        while self.c_tree is not None:
            if self.c_tree == tree:
                self.c_tree = None
            self.c_tree = self.c_tree.neighbor

    # input: star tree and star tree root
    # implements the star cubing algorithm
    def starcubing(self, t, cnode):
        cc = None
        separator = ':'

        # while in the base tree, insert or aggregate the current node to each of its neighboring trees
        # additional printing is added for debugging purposes if the node was skipped
        self.c_tree = self.base_tree
        while self.c_tree.neighbor is not None:
            print(
                separator.join([str(self.c_tree.neighbor.root.get_data()), str(self.c_tree.neighbor.root.get_count())]),
                end='\t')
            if self.c_tree.neighbor.head.depth > cnode.depth:
                print("start traversing up. ", end='')
                while self.c_tree.neighbor.head.get_parent() is not None \
                        and self.c_tree.neighbor.head.get_data().split('_')[0] != cnode.get_data().split('_')[0]:
                    self.c_tree.neighbor.head = self.c_tree.neighbor.head.get_parent()
                    print(self.c_tree.neighbor.head.data, end='--')
                print("end traversal.")
                print()
                print("insert right: ", end='')
                # print(separator.join([str(self.c_tree.neighbor.root.get_data()), str(self.c_tree.neighbor.root.get_count())]), end = '\t')
                if self.c_tree.neighbor.head.depth == cnode.depth:
                    if cnode.data == self.c_tree.neighbor.head.data or self.getpath(
                            cnode) == self.c_tree.neighbor.head.data:
                        self.c_tree.neighbor.head.set_count(self.c_tree.neighbor.head.get_count() + cnode.get_count())
                        print("aggregated ", end='')
                        print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                    else:
                        new_node = Node(cnode.data, cnode.count, parent=self.c_tree.neighbor.head.parent,
                                        depth=cnode.depth)
                        self.c_tree.neighbor.insert_right(new_node)
                        print("inserted ", end='')
                        print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                else:
                    print("skipped: ", end='')
                    print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
            else:
                print("insert left: ", end='')
                if self.c_tree.neighbor.head.get_left() is not None:
                    temp = self.c_tree.neighbor.head.get_left()
                    while temp is not None and temp.get_data() != cnode.get_data():
                        temp = temp.get_right()
                    if temp is not None and temp.get_data() == cnode.get_data():
                        self.c_tree.neighbor.head = temp
                        self.c_tree.neighbor.head.set_count(self.c_tree.neighbor.head.get_count() + cnode.get_count())
                        print("aggregated ", end='')
                        print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                    elif temp is not None and temp.get_parent() is not None:
                        new_node = Node(cnode.data, cnode.count, parent=self.c_tree.neighbor.head.parent,
                                        depth=cnode.depth)
                        self.c_tree.neighbor.insert_right(new_node)
                        print("inserted right: ", end='')
                        print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                    else:
                        new_node = Node(cnode.data, cnode.count, parent=self.c_tree.neighbor.head, depth=cnode.depth)
                        self.c_tree.neighbor.insert_left(new_node)
                        print("inserted ", end='')
                        print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                elif self.c_tree.neighbor.head.depth == cnode.depth - 1 or cnode.get_parent().data is None:
                    new_node = Node(cnode.data, cnode.count, parent=self.c_tree.neighbor.head, depth=cnode.depth)
                    self.c_tree.neighbor.insert_left(new_node)
                    print("inserted ", end='')
                    print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                else:
                    print("skipped: ", end='')
                    print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
            self.c_tree = self.c_tree.neighbor
        print()

        # if count of cnode meets threshold, outputs to file
        # if cnode is not a leaf, create a new neighboring tree and append to the last neighbor
        if cnode.get_count() >= self.min_sup:
            if cnode != root:
                f.write(separator.join([str(cnode.get_data()), str(cnode.get_count())]))
                f.write(",")
                printtemp = self.getpathprint(cnode)
                f.write(printtemp)
                f.write("\n")
            if cnode.get_left() is None:
                self.printpath(cnode)
            else:
                cc = Node(self.getpath(cnode), cnode.get_count(), depth=cnode.depth)
                tc = Tree(cc)
                self.insert_neighbor(tc)

                print("neighbor inserted: ", end='')
                print(separator.join([str(cnode.get_data()), str(cnode.get_count())]))

        # recursion
        if cnode.get_left() is not None:
            self.starcubing(t, cnode.get_left())
        # delete the child once all children of the child tree have been processed
        if cc is not None:
            print("Erasing child tree:", end=' ')
            print(tc.root.data)
            print()
            tc.parent.neighbor = None
            print("Erasing child node:", end=' ')
            print(cnode.data)
            print()
            cnode.set_left(None)
        # recursion for siblings
        if cnode.get_right() is not None:
            self.starcubing(t, cnode.get_right())

    # gets the path to the current node
    def getpath(self, node):
        temp = node
        result = []
        separator = ''
        while temp is not None:
            result.insert(0, temp.get_data())
            temp = temp.get_parent()
        return separator.join(result)

    # gets the path to the current node for output file with comma delimiter
    def getpathprint(self, node):
        temp = node
        result = []
        separator = ','
        while temp is not None:
            result.insert(0, temp.get_data())
            temp = temp.get_parent()
        return separator.join(result)

    # directly prints the path to the current node
    def printpath(self, node):
        temp = node
        separator = ':'
        while temp is not None:
            print(separator.join([temp.get_data(), str(temp.get_count())]), end="\t")
            temp = temp.get_parent()
        print()

    # prints current node and its children
    def print2DUtil(self, root):
        separator = ':'
        if (root == None):
            return

        print(separator.join([str(root.get_data()), str(root.get_count())]), end='\t')
        if root.get_left() is not None:
            root = root.get_left()
            print(separator.join([str(root.get_data()), str(root.get_count())]), end='\t')
            while root.get_right() is not None:
                root = root.get_right()
                print(separator.join([str(root.get_data()), str(root.get_count())]), end='\t')
        else:
            print("Leaf: ", end='')
            print(separator.join([str(root.get_data()), str(root.get_count())]), end='\t')
        print()


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
