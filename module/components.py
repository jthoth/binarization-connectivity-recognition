from itertools import product
from . import unionfind


class Components(object):

    def __init__(self, image):
        self.image, self.component = image, dict()

    def __update__(self, _current, _next):
        """
         update component values
        :param _current:
        :param _next:
        :return:
        """
        self.component.__setitem__(
            _current, self.component.get(_next)
        )

    def __create_label__(self, _current, union):
        """
            given a init point and a union-find structure this will create
            a new label
        :param _current: init point
        :param union: data-structure
        :return:
        """
        self.component.__setitem__(
            _current, union.generate_label()
        )

    def __union_between_components(self, h, w, _union):
        top_right = self.component.get((h - 1, w + 1))
        self.component.__setitem__((h, w), top_right)
        if w and not self.image[h - 1, w - 1]:
            _union.union(top_right, self.component[(h - 1, w - 1)])
        elif w and not self.image[h, w - 1]:
            _union.union(top_right, self.component[(h, w - 1)])

    @staticmethod
    def _inverse(union, components):
        """

        :param union:
        :param components:
        :return:
        """

        component_coordinates = dict()
        for coordinates, value in components.items():
            component = union.find(value)
            container = component_coordinates.get(component)
            if container:
                container.append(coordinates)
            else:
                component_coordinates.__setitem__(component, [coordinates])

        return component_coordinates

    def get(self):
        """
        This method use the two-label components
        :return: components dict()
        """
        (height, width) = self.image.shape
        union = unionfind.UnionFind()
        for h, w in product(range(height), range(width)):
            if self.image[h, w]:
                continue
            elif h and not self.image[h - 1, w]:
                self.__update__((h, w), (h - 1, w))
            elif w + 1 < width and h and not self.image[h - 1, w + 1]:
                self.__union_between_components(h, w, union)
            elif h * w and not self.image[h - 1, w - 1]:
                self.__update__((h, w), (h - 1, w - 1))
            elif w and not self.image[h, w - 1]:
                self.__update__((h, w), (h, w - 1))
            else:
                self.__create_label__((h, w), union)

        return self._inverse(union, self.component)
