import cv2
import numpy as np

from scipy.ndimage import rotate
from scipy.stats import bernoulli

import util

POST_PROCESSING_SIZE = 65

FIRST_ITERATION = 'driving/'
SECOND_ITERATION = 'driving2/'
THIRD_ITERATION = 'not/track1_recovery/'
ERRORS_ITERATION = 'errors/'
INITIAL_ITERATION = 'data/'
SELECTED_FOLDER = INITIAL_ITERATION
RANGE = 220
FAIR_COIN = 0.5
UNFAIR_COIN = 0.9
MAX_ROTATION = 16


def flip_a_coin(probability=FAIR_COIN):
    return bernoulli.rvs(probability)


def random_gamma(image):
    """
    Used gamma correction suggested here: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    gamma = np.random.uniform(0.3, 1.8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def random_rotation(image, initial_angle):
    angle = np.random.uniform(-MAX_ROTATION, MAX_ROTATION + 1)
    return rotate(image, angle, reshape=False), initial_angle + (-1) * (np.pi / 180.0) * angle


def shear(image, angle):
    """
    Idea taken from the article:
    https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    """

    if not flip_a_coin(UNFAIR_COIN):
        return image, angle
    y, x, channel = image.shape

    delta_x = np.random.randint(-RANGE, RANGE + 1)
    pts1 = np.float32([[0, y], [x, y], [x / 2, y / 2]])
    pts2 = np.float32([[0, y], [x, y], [x / 2 + delta_x, y / 2]])
    amendment = delta_x / (y / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    image = cv2.warpAffine(image, cv2.getAffineTransform(pts1, pts2), (x, y), borderMode=1)
    angle += amendment

    return image, angle


def pre_processing(filename, mirrored, angle):
    image = cv2.imread(INITIAL_ITERATION + filename)
    if mirrored:
        image = np.fliplr(image)

    image, angle = shear(image, angle)
    image = util.crop(image)
    image = random_gamma(image)
    image = cv2.resize(image, (POST_PROCESSING_SIZE, POST_PROCESSING_SIZE))

    return image, angle
