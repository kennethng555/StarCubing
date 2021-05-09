from Node import Node


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
                if self.head.get_left() == None:
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
