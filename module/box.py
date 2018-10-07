import numpy as np
from skimage.draw import polygon_perimeter


class BoundBox(object):

    def __init__(self, image,  components, exclude=(25, 100), padding=1):
        """

        :param image:
        :param components:
        :param exclude:
        :param padding:
        """
        self.components_coordinates = dict()
        self.__constrains(components)
        self.padding = self.padding_structure(padding)
        self.components = components.copy()
        self.exclude = exclude
        self.image = image

    @staticmethod
    def padding_structure(padding):
        _kwargs = dict(pad_width=padding)
        _kwargs.__setitem__('constant_values', 1)
        _kwargs.__setitem__('mode', 'constant')
        return _kwargs

    @staticmethod
    def __constrains(components):
        if not isinstance(components, dict):
            raise ValueError('The component should by a dict')

    def __update_component(self, label,  component):
        percentiles = np.percentile(component, (0, 100), axis=0)
        (min_y, min_x), (max_y, max_x) = percentiles.astype(int)
        bound_box = self.image[min_y:max_y, min_x:max_x]
        bound_box = np.pad(bound_box, **self.padding)
        self.components.__setitem__(label, bound_box)
        self.components_coordinates.__setitem__(
            label, ((min_y, min_x), (max_y, max_x))
        )

    def __clean_outliers(self, excluded):
        _filter = filter(self.components.__contains__, excluded)
        list(map(self.components.__delitem__, _filter))

    def __order_sequential(self):
        """
        Sometimes values get one label is not ordered
        for example the input is 123 and the algorithm
        component get the 2 as the first label this method
        fix that
        :return:
        """

        def get_ordered_value(value):
            """
            this method get the first element and the
            ming value of a dict
            :param value: key and coordinates
            :return:
            """

            return value[1][0][1]

        ordered_components = dict(sorted(
            self.components_coordinates.items(), key=get_ordered_value
        ))

        template = dict()

        for i, key in enumerate(ordered_components.keys()):
            template.__setitem__(i, self.components.get(key))

        self.components = template

    def get(self):

        component_dimensions = list(map(len, self.components.values()))
        boundary, _ = np.percentile(component_dimensions, self.exclude)
        excluded = list()

        for label, component in self.components.items():
            if component.__len__() > boundary:
                self.__update_component(label, component)
            else:
                excluded.append(label)

        self.__clean_outliers(excluded)
        self.__order_sequential()

        return self.components


class BoundBoxDraw(object):

    def __init__(self, image, components_coordinates, pad=2):
        self.coordinates = components_coordinates
        self.image = image
        self.pad = pad

    @staticmethod
    def rectangle_perimeter(min_y, min_x, width, height):
        """

        :param min_y: min value of y
        :param min_x: min value of x
        :param height: compute max value of y minus min y
        :param width: compute max value of x minus min x
        :return:
        """
        rr, cc = (
            [min_y, min_y + width, min_y + width, min_y],
            [min_x, min_x, min_x + height, min_x + height]
        )

        return polygon_perimeter(rr, cc)

    def draw(self, color):
        for (min_y, min_x), (max_y, max_x) in self.coordinates:
            rr, cc = self.rectangle_perimeter(
                min_y - self.pad, min_x - self.pad,
                (max_y + self.pad - min_y), (max_x + self.pad - min_x)
            )
            self.image[rr, cc] = color
        return self.image
