import os
import argparse
import numpy as np
from time import time

from skimage import io, color, img_as_float
from scipy import ndimage
from module import threshold
from module import components
from module import contour
from module import box


class Program(object):

    def __init__(self, args):
        self.color_image = None
        self.args = args

    def load_threshold_sample(self):
        image = io.imread(self.args.input)
        self.color_image = image
        _image = color.rgb2gray(image)
        _image = ndimage.uniform_filter(_image, size=self.args.blur)
        obj_threshold = threshold.Threshold(_image)
        boundary = obj_threshold.compute_global_otsu()
        return (_image > boundary).astype(np.uint8)

    @staticmethod
    def get_components(image):
        connectivity_searcher = components.Components(image)
        return connectivity_searcher.get()

    def save_bound_box_detected(self, bounder_box):
        bound_box_drawer = box.BoundBoxDraw(
            self.color_image, bounder_box.components_coordinates.values()
        )
        image = bound_box_drawer.draw([255, 0, 0])
        io.imsave(self.args.output_image, image)

    @staticmethod
    def concat_images(img_a, img_b):

        ha, wa = img_a.shape[: 2]
        hb, wb = img_b.shape[: 2]

        max_height = np.max([ha, hb])
        total_width = wa + wb

        new_img = np.ones(shape=(max_height, total_width))
        new_img[: ha, : wa] = img_a
        new_img[: hb, wa: wa + wb] = img_b

        return new_img

    def save_contour(self, c_components):
        contour_bound = np.ones(shape=(1, 1), dtype=np.uint8)
        for _components in c_components.values():
            new_image = np.ones(shape=_components.shape)
            _contour = contour.ContourDetection(_components)
            for coordinates in _contour.get():
                new_image[coordinates] = 0
            contour_bound = self.concat_images(contour_bound, new_image)
        image = img_as_float(contour_bound)
        io.imsave(self.args.output_contour, image)

    def get_features_from_samples(self, root, files):
        """

        :param root:
        :param files:
        :return:
        """
        sample_features = list()
        for image in files:
            file_name = '{}{}{}'.format(root, os.sep, image)
            _image = io.imread(file_name, as_gray=True)
            _image = ndimage.uniform_filter(_image, size=self.args.blur)
            obj_threshold = threshold.Threshold(_image)
            boundary = obj_threshold.compute_global_otsu()
            _image = (_image > boundary).astype(np.uint8)
            _image = np.pad(_image, pad_width=self.args.pad,
                            mode='constant', constant_values=1)
            _contour = contour.ContourDetection(_image)
            sample_features.append(_contour.get_features())

        return np.array(sample_features).mean(axis=0)

    def load_samples(self):
        init = time()
        print('Loading samples to learn applying the mean....')
        label, classes = list(), dict()

        for i, (root, dirs, files) in enumerate(os.walk(self.args.samples)):
            if dirs:
                label = dirs
            else:
                classes.__setitem__(
                    label[i - 1], self.get_features_from_samples(root, files)
                )
        if len(classes) < 1:
            raise ValueError('The algorithm need samples to predict the input')
        print('Loaded.... time executed : {} ms'.format(time() - init))
        return classes

    def save_prediction(self, result):
        with open(self.args.output_file, 'w') as file:
            file.write(result)

    def predict(self, c_components):
        """

        :param c_components:
        :return:
        """
        sample_cases = self.load_samples()
        rut_output, labels = str(), list(sample_cases.keys())
        for one_component in c_components.values():
            _contour = contour.ContourDetection(one_component)
            features = _contour.get_features()
            distances = [np.linalg.norm(features - samples_prediction)
                         for samples_prediction in sample_cases.values()
                         ]
            rut_output += str(labels[int(np.argmin(distances))])
        print('Rut computed: {}'.format(rut_output))
        self.save_prediction(rut_output)

    def run(self):
        init = time()
        print('Initializing....')

        image = self.load_threshold_sample()
        c_components = self.get_components(image)
        bounder_box = box.BoundBox(
            image, c_components, padding=self.args.pad
        )

        connected_components = bounder_box.get()
        self.predict(connected_components)

        self.save_contour(connected_components)
        self.save_bound_box_detected(bounder_box)

        print('Predicting.... time executed : {} ms'.format(time() - init))


def main():
    """

    :return:
    """

    parser = argparse.ArgumentParser(
        description='Threshold connected components and prediction using python and numpy strategies'
    )
    parser.add_argument('-oi', '--output_image', help='Output image bounding box',
                        default='result/bounding_box.jpg'
                        )
    parser.add_argument('-oc', '--output_contour', help='Output image contour ',
                        default='result/contour.jpg'
                        )
    parser.add_argument('-of', '--output_file', help='Output file name with prediction',
                        default='result/prediction.txt'
                        )

    parser.add_argument('-b', '--blur', default=2, type=int)
    parser.add_argument('-p', '--pad', default=1, type=int)
    parser.add_argument('-s', '--samples', default='modelsv2')

    required_args = parser.add_argument_group('required named arguments')
    required_args.add_argument(
        '-i', '--input', help='Input file name path', required=True
    )
    args = parser.parse_args()

    runner = Program(args)
    runner.run()


if __name__ == '__main__':
    main()
