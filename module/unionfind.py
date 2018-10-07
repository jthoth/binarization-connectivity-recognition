"""
 Custom Implementation of union find.

"""


class UnionFind(object):

    """
        This class generate a label vertex connection given in algorithm class

    """

    def __init__(self):
        """
         Constructor of python class

        """

        self.P = list()
        self.label = 0

    def generate_label(self):
        """

        This method produce a new label

        :return: return key of the label

        """

        last = self.label
        self.label += 1
        self.P.append(last)

        return last

    def set_root(self, x, root):
        """
        This method set a element as root of the structure
        :param x: x-element
        :param root: old-root
        :return: None
        """

        while self.P[x] < x:

            j = self.P[x]
            self.P[x] = root
            x = j

        self.P[x] = root

    def find_root(self, i):
        """

        :param i:
        :return:
        """

        while self.P[i] < i:
            i = self.P[i]

        return i

    def find(self, x):
        """
         search an element into the structure

        :param x:
        :return: the root of the structure
        """
        root = self.find_root(x)
        self.set_root(x, root)

        return root

    def union(self, x, y):
        """

        :param x: x-element constructed
        :param y: y-element constructed
        :return: None
        """

        if x != y:

            x_root = self.find_root(x)
            y_root = self.find_root(y)

            maximum = (y_root if x_root > y_root
                       else x_root)

            self.set_root(x, maximum)
            self.set_root(y, maximum)
