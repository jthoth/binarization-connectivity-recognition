import unittest
from module import threshold
from skimage import io


class ThresholdTest(unittest.TestCase):

    def setUp(self):
        image = io.imread('samples/rut_2.jpg', as_gray=True)
        self.experiment = threshold.Threshold(image)

    def test_compare_otsu(self):
        result = self.experiment.compute_global_otsu()
        expected = self.experiment.compute_global_api_otsu()
        self.assertEqual(expected, result)
