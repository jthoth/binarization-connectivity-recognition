from itertools import product
import numpy as np


class ContourDetection(object):
    """
    Search a order contour from a binary image
    """

    def __init__(self, component):
        """

        :param component:
        """
        self.neighbours = self.get_neighbours_mesh()
        self.component = component
        self.__features = list()

    @staticmethod
    def padding_structure(padding):
        _kwargs = dict(pad_width=padding)
        _kwargs.__setitem__('constant_values', 1)
        _kwargs.__setitem__('mode', 'constant')
        return _kwargs

    @staticmethod
    def get_neighbours_mesh():
        """
        building the mesh-grid 8 neighbours
        :return:
        """
        return ((0, -1), (-1, -1), (-1, 0), (-1, 1),
                (0, 1), (1, 1), (1, 0), (1, -1))

    def _next_inner_point(self, h, w):
        """

        :param h: current position in y
        :param w: current position in w
        :return: index neighbours selected and next-point
        """
        for _index, (h_w, w_w) in enumerate(self.neighbours):
            if not self.component[(h + h_w, w + w_w)]:
                return _index, (h + h_w, w + w_w)

    def set_feature_direction(self, _index):
        """

        :param _index:
        :return:
        """
        h, w = self.neighbours[_index]
        self.__features.append(h * 3 + w)

    def contour_ordered_computation(self, init_points):
        """
        compute the sequential contour using clock-wise methodology
        :param init_points: first point or p0
        :return:
        """
        _image_contour_coord, _index, infinite = list(), 0, 0
        _image_contour_coord.append(init_points)
        h, w = init_points

        while True and infinite < 100000:
            _index, (h, w) = self._next_inner_point(h, w)
            self.set_feature_direction(_index)
            self.neighbours = (self.neighbours[_index - 1:] +
                               self.neighbours[:_index - 1])
            _image_contour_coord.append((h, w))
            infinite += 1
            if _image_contour_coord[0] == (h, w):
                break

        return _image_contour_coord

    def get(self):
        """
        :return: Contour
        """
        height, width = self.component.shape

        for h, w in product(range(height), range(width)):
            if self.component[h, w]:
                continue
            else:
                return self.contour_ordered_computation(
                    (h, w)
                )

    def get_features(self, normed=True):
        """

        :return:
        """

        self.get()

        features, bins = np.histogram(
            self.__features, bins=8, range=(-4, 5), density=normed
        )

        return features

    def get_split_features(self, size=4, normed=True):
        """

        :return:
        """
        block_features = np.array([])
        height, width = self.component.shape
        m, n = height // (size // 2), width // (size // 2)

        components = [self.component[x - m: x, y - n: y]
                      for x in range(m, height + 1, m)
                      for y in range(n, width + 1, n)
                      ]

        for block in components:
            self.component = np.pad(block, **self.padding_structure(1))
            self.get()
            features, bins = np.histogram(
                self.__features, bins=8, range=(-4, 5), density=normed
            )
            block_features = np.append(block_features, features)

        return block_features
