import numpy as np

CROP = [0.337, 0.1]


def crop(image):
    top = int(np.ceil(CROP[0] * image.shape[0]))
    bottom = image.shape[0] - int(np.ceil(CROP[1] * image.shape[0]))

    return image[top:bottom, :]
