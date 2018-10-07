import numpy as np
from skimage import filters


class Threshold(object):
    """
        This class had implemented the required from task #1:
        applying:
            - Otsu coded by hand
            - adaptive threshold using a function library
    """

    def __init__(self, image, n_bins=256):
        self.image = image
        self.n_bins = n_bins

    @staticmethod
    def ignore_last(array):
        """
                One-dimention array
        :param array:
        :return:
        """
        return array[:-1]

    @staticmethod
    def ignore_first(array):
        """
               One-dimention array
        :param array:
        :return:
        """
        return array[1:]

    def __compute_histogram(self):
        """
           Compute the histogram of an image and fix the edges bin

        :return:
        """
        histogram, bin_edges = np.histogram(
            self.image.flat, bins=self.n_bins
        )

        centroids = (self.ignore_first(bin_edges) + self.ignore_last(bin_edges)) / 2.
        return histogram.astype(np.float32), centroids

    @staticmethod
    def __compute_prob_weights(histogram):
        """
        Compute the probabilistic weights
        :param histogram:
        :return:
        """

        return np.cumsum(histogram), np.cumsum(np.flipud(histogram))

    @staticmethod
    def __compute_medians(non_normalized_histogram, weight_one, weight_two):
        mean_one = np.cumsum(non_normalized_histogram) / weight_one
        mean_two = np.cumsum(np.flipud(non_normalized_histogram)) / weight_two
        return mean_one, mean_two

    def compute_global_otsu(self):

        """
            Own Implementation using Numpy
        :return:
        """

        histogram, centroids = self.__compute_histogram()
        weight_one, weight_two = self.__compute_prob_weights(histogram)
        non_normalized_histogram = histogram * centroids
        mean_one, mean_two = self.__compute_medians(
            non_normalized_histogram, weight_one, weight_two
        )
        variance = weight_one*np.flipud(weight_two)*(mean_one - np.flipud(mean_two))**2
        index = np.argmax(variance)

        return centroids[index]

    def compute_global_api_otsu(self):
        return filters.thresholding.threshold_otsu(self.image)

    def compute_local_api(self, block_size, offset=0):
        return filters.threshold_local(self.image, block_size, offset=offset)
