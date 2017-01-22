import cv2
import numpy as np

from scipy.ndimage import rotate
from scipy.stats import bernoulli

"""
Defined constants
"""
POST_PROCESSING_SIZE = 65
FIRST_ITERATION = 'driving/'
SECOND_ITERATION = 'driving2/'
THIRD_ITERATION = 'not/track1_recovery/'
ERRORS_ITERATION = 'errors/'
INITIAL_ITERATION = 'data/'
SELECTED_FOLDER = INITIAL_ITERATION
CROP = [0.337, 0.1]
RANGE = 220
FAIR_COIN = 0.5
UNFAIR_COIN = 0.9
MAX_ROTATION = 16


def flip_a_coin(probability=FAIR_COIN):
    """
    Flips a coin with probability
    :param probability:
        The probability of heads
    :return:
        True or false
    """
    return bernoulli.rvs(probability)


def crop(image):
    """
    Crop image based on cropping rates defined above
    :param image:
        Image to crop
    :return:
        Cropped image
    """
    top = int(np.ceil(CROP[0] * image.shape[0]))
    bottom = image.shape[0] - int(np.ceil(CROP[1] * image.shape[0]))

    return image[top:bottom, :]


def random_gamma(image):
    """
    Used gamma correction suggested here: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    :param image:
        Image to adjust gamma to
    :return:
        Adjusted image
    """
    gamma = np.random.uniform(0.3, 1.8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def random_rotation(image, initial_angle):
    """
    Add random rotation in the predefined brackets
    :param image:
        Image to adjust
    :param initial_angle:
        Angle to adjust
    :return:
        Tuple of adjusted image and new calculated corresponding angle
    """
    angle = np.random.uniform(-MAX_ROTATION, MAX_ROTATION + 1)
    return rotate(image, angle, reshape=False), initial_angle + (-1) * (np.pi / 180.0) * angle


def shear(image, angle):
    """
    Add shear to the image and it's angle. Idea taken from the article:
    https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    :param image:
        Image to adjust
    :param angle:
        Angle to adjust
    :return:
        Tuple of adjusted image and new calculated corresponding angle
    """
    # probability of running the shear change
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
    """
    Loads image and init preprocessing
    :param filename:
        File name of the image to open
    :param mirrored:
        True if the image should be flipped
    :param angle:
        corresponding angle (label)
    :return:
        Tuple of adjusted feature and label (image and angle)
    """
    image = cv2.imread(INITIAL_ITERATION + filename)
    if mirrored:
        image = np.fliplr(image)

    image, angle = shear(image, angle)
    image = crop(image)
    image = random_gamma(image)
    image = cv2.resize(image, (POST_PROCESSING_SIZE, POST_PROCESSING_SIZE))

    return image, angle
