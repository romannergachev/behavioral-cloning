import numpy as np
import math
import csv
import cv2
import json

from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
import tensorflow as tf

EPOCHS = 5
BATCH = 100
WIDTH = 160
HEIGHTS = 80
CHANNEL_NUMBER = 3
REG = 0.01
FINE_TUNING = True

SHAPE = (HEIGHTS, WIDTH, CHANNEL_NUMBER)
tf.python.control_flow_ops = tf


def learning_rate():
    if FINE_TUNING:
        return 0.00001
    else:
        return 0.001


def apply_clahe2(filename, mirrored=False):
    image = cv2.imread('data/' + filename)
    image = cv2.resize(image, (WIDTH, HEIGHTS))
    if mirrored:
        image = cv2.flip(image, 1)
    img = image

    blue = img[:, :, 0]
    red = img[:, :, 2]
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    claheObj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    clahe = claheObj.apply(gray_image)
    claheRed = claheObj.apply(red)
    claheBlue = claheObj.apply(blue)

    return np.stack((claheBlue, claheRed, clahe), axis=2)


def apply_clahe(filename, mirrored=False):
    image = cv2.imread('data/' + filename)
    image = cv2.resize(image, (WIDTH, HEIGHTS))
    if mirrored:
        image = cv2.flip(image, 1)

    img = image[np.newaxis, ...]
    return img


def generate_cnn_model():
    cnn = Sequential()
    cnn.add(Lambda(lambda x: x / 128. - 1., input_shape=SHAPE, output_shape=SHAPE))
    cnn.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=SHAPE, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(Convolution2D(36, 5, 5, subsample=(2, 2), W_regularizer=l2(REG)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(Convolution2D(48, 5, 5, subsample=(2, 2), W_regularizer=l2(REG)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(Convolution2D(64, 3, 3, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(Convolution2D(64, 3, 3, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(Flatten())
    cnn.add(Dense(100, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(50, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(10, W_regularizer=l2(REG), b_regularizer=l2(REG)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(1, W_regularizer=l2(REG), b_regularizer=l2(REG)))

    return cnn


def generate_data_from_driving(images):
    closed_counter = 0
    while 1:
        final_images = np.ndarray((BATCH, HEIGHTS, WIDTH, CHANNEL_NUMBER), dtype=float)
        final_angles = np.ndarray(BATCH, dtype=float)
        for k in range(BATCH):
            if closed_counter == len(images):
                shuffle(images)
                closed_counter = 0

            filename = images[closed_counter][0]
            angle = images[closed_counter][1]
            mirrored = images[closed_counter][2]
            final_image = apply_clahe(filename, mirrored)
            final_angle = np.ndarray(1, dtype=float)
            final_angle[0] = angle
            final_images[k] = final_image
            final_angles[k] = angle
            closed_counter += 1
        yield ({'lambda_input_1': final_images}, {'dense_4': final_angles})


def epoch_data_length(data_length):
    return math.ceil(data_length / BATCH) * BATCH


with open('data/driving_log.csv', 'r') as file:
    reader = csv.reader(file)
    driving_log = list(reader)

driving_log_length = len(driving_log)

X_train = [("", 0.0, False) for x in range(driving_log_length)]

for i in range(driving_log_length):
    X_train[i] = (driving_log[i][0].lstrip(), float(driving_log[i][3]), 0)

driving_log_length = len(X_train)

if not FINE_TUNING:
    for i in range(driving_log_length):
        if X_train[i][1] != 0.0:
            X_train.append([X_train[i][0], -1.0 * X_train[i][1], True])

driving_log_length = len(X_train)

X_train = shuffle(X_train)

train_elements_len = int(3.0 * driving_log_length / 4.0)
valid_elements_len = int(driving_log_length / 4.0 / 2.0)

X_valid = X_train[train_elements_len:train_elements_len + valid_elements_len]
X_test = X_train[train_elements_len + valid_elements_len:]
X_train = X_train[:train_elements_len]

if FINE_TUNING:
    with open("model.json", 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
else:
    model = generate_cnn_model()

model.compile(Adam(learning_rate), 'mean_squared_error', ['accuracy'])

history = model.fit_generator(
    generate_data_from_driving(X_train),
    epoch_data_length(len(X_train)),
    EPOCHS,
    validation_data=generate_data_from_driving(X_valid),
    nb_val_samples=epoch_data_length(len(X_valid)),
    max_q_size=10
)

accuracy = model.evaluate_generator(generate_data_from_driving(X_test), epoch_data_length(len(X_test)))

print("Test accuracy {}".format(accuracy))

model.save_weights("./model.h5")

json_data = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(json_data, json_file)

print("Saved model to disk")
