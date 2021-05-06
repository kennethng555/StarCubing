from Node import Node
from Tree import Tree


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
        f = open("../output.txt", "w+")
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
            if cnode != t.root:
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

        f.close()

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

