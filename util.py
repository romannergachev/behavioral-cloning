import numpy as np

CROP = [0.337, 0.112]


def crop(image):
    top = int(np.ceil(image.shape[0] * CROP[0]))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * CROP[1]))

    return image[top:bottom, :]
