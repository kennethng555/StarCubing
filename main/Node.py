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