# Load pickled data
from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
import pickle
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import collections
import cv2

from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

EPOCHS = 5
WIDTH = 160
HEIGHTS = 80
LEARNING_RATE = 0.001
FINE_TUNING = False
CHANNEL_NUMBER = 1
REGULARIZATION = 0.001

INPUT_SHAPE = (WIDTH, HEIGHTS, CHANNEL_NUMBER)

def signSpecifiedClahe(img):
    blue = img[:, :, 0]
    red = img[:, :, 2]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    claheObj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    clahe = claheObj.apply(gray_image)
    claheRed = claheObj.apply(red)
    claheBlue = claheObj.apply(blue)

    #return np.stack((claheBlue, claheRed, clahe), axis=2)
    return clahe

class DLProgress(tqdm):
    last_block = 0

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')

with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train = data['features']
y_train = data['labels']

X_train, y_train = shuffle(X_train, y_train)

def normalize_grayscale(image_data):
    image_data = signSpecifiedClahe(image_data)
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min)*(b - a))/(grayscale_max - grayscale_min))

X_normalized = normalize_grayscale(X_train)
resized = tf.image.resize_images(X_normalized, [WIDTH, HEIGHTS])


label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model = Sequential()
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=INPUT_SHAPE,  W_regularizer=l2(REGULARIZATION)))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), W_regularizer=l2(REGULARIZATION)))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), W_regularizer=l2(REGULARIZATION)))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(REGULARIZATION)))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(REGULARIZATION)))
model.add(Dropout(0.5))
model.add(LeakyReLU())
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(REGULARIZATION)))
model.add(LeakyReLU())
model.add(Dense(50, W_regularizer=l2(REGULARIZATION)))
model.add(LeakyReLU())
model.add(Dense(10, W_regularizer=l2(REGULARIZATION)))
model.add(LeakyReLU())
model.add(Dense(1, W_regularizer=l2(REGULARIZATION)))
#model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=EPOCHS, validation_split=0.2)

with open('test.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# TODO: Preprocess data & one-hot encode the labels
X_normalized_test = normalize_grayscale(X_test)
y_one_hot_test = label_binarizer.fit_transform(y_test)

# TODO: Evaluate model on test data
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))