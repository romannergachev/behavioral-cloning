import numpy as np
import math
import csv
import json

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import MaxPooling2D
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import tensorflow as tf
import preprocessing

"""
Defined constants
"""
EPOCHS = 15
BATCH = 64
CHANNEL_NUMBER = 3
ANGLE_COEFFICIENT = 0.22
FINE_TUNING = False
CHOICE_LIST = ['CENTER', 'LEFT', 'RIGHT']
SHAPE = (preprocessing.POST_PROCESSING_SIZE, preprocessing.POST_PROCESSING_SIZE, CHANNEL_NUMBER)

tf.python.control_flow_ops = tf


def learning_rate():
    """
    Selects the learning rate based on fine tuning
    :return:
        Selected learning rate
    """
    if FINE_TUNING:
        return 0.000001
    else:
        return 0.0001


def generate_cnn_model():
    """
    Generates the simple cnn model as described in Nvidia article:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Leaky relus are used instead of the basic ones since they allows a small gradient when the unit is not active -
    makes number of dead "neurons" smaller
    :return:
        Generated model
    """
    cnn = Sequential()
    cnn.add(Lambda(lambda x: x / 127.5 - 1., input_shape=SHAPE, output_shape=SHAPE))
    cnn.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=SHAPE))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.5))
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Flatten())
    cnn.add(Dense(100))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(Dense(50))
    cnn.add(LeakyReLU())
    cnn.add(Dense(10))
    cnn.add(LeakyReLU())
    cnn.add(Dense(1))
    cnn.summary()
    return cnn


def generate_data_from_driving(data):
    """
    Generates data by portions, in order not to load all the images in memory at once.
    Yields generated features (images) and labels (steering angles) to the model.
    :param data:
        data from csv file containing images file names and steering angle
    :return:
        prepared features and labels for the model
    """
    # counter for the infinite loop (closed on the data quantity), starting from 0 as soon as it gets to len(data)
    closed_counter = 0
    while 1:
        x = np.ndarray((BATCH, *SHAPE), dtype=float)
        y = np.ndarray(BATCH, dtype=float)
        for k in range(BATCH):
            if closed_counter == len(data):
                shuffle(data)
                closed_counter = 0

            # randomly select center, left or right image
            random_image = np.random.choice(CHOICE_LIST)

            # flipping a coin to decide whether to mirror the image or not
            mirrored = preprocessing.flip_a_coin()

            if random_image == CHOICE_LIST[0]:
                filename = data[closed_counter][0]
                angle = data[closed_counter][3]
                if mirrored:
                    angle = -angle
            elif random_image == CHOICE_LIST[1]:
                filename = data[closed_counter][1]
                angle = data[closed_counter][3] + ANGLE_COEFFICIENT
                if mirrored:
                    angle = -angle
            else:
                filename = data[closed_counter][2]
                angle = data[closed_counter][3] - ANGLE_COEFFICIENT
                if mirrored:
                    angle = -angle

            # initiate image preprocessing
            x[k], y[k] = preprocessing.pre_processing(filename, mirrored, angle)
            closed_counter += 1
        yield ({'lambda_input_1': x}, {'dense_4': y})


def epoch_data_length(data_length):
    """
    Return the amount of data directly proportional to the batch size
    :param data_length:
        length of data
    :return:
        length of sample directly proportional to the data length
    """
    return math.ceil(data_length / BATCH) * BATCH

# read the driving_log.csv
with open(preprocessing.SELECTED_FOLDER + 'driving_log.csv', 'r') as file:
    reader = csv.reader(file)
    driving_log = list(reader)

driving_log_length = len(driving_log)

# vector to store image names and corresponding angle
Xy_train = [("", "", "", 0.0) for x in range(driving_log_length)]

# save data from .csv to the vector
for i in range(driving_log_length):
    Xy_train[i] = (
        driving_log[i][0].lstrip(),
        driving_log[i][1].lstrip(),
        driving_log[i][2].lstrip(),
        float(driving_log[i][3])
    )

driving_log_length = len(Xy_train)

# randomize the data distribution
Xy_train = shuffle(Xy_train)

# take 80% of data as train data, divide the rest amount between validation and testing sets
train_elements_len = int(4.0 * driving_log_length / 5.0)
valid_elements_len = int(driving_log_length / 5.0 / 2.0)

Xy_valid = Xy_train[train_elements_len:train_elements_len + valid_elements_len]
Xy_test = Xy_train[train_elements_len + valid_elements_len:]
Xy_train = Xy_train[:train_elements_len]

# if fine tuning - load model from file, otherwise create new model
if FINE_TUNING:
    with open("model.json", 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)
else:
    model = generate_cnn_model()

# adam optimizer with MSE to track
model.compile(Adam(learning_rate), 'mse')

"""
init keras checkpoint callback and early stopping in order to give the opportunity to stop the model when it is not
improving anymore and checkpoint - to save best weights
"""
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=2, verbose=1)

# train the model
history = model.fit_generator(
    generate_data_from_driving(Xy_train),
    3 * epoch_data_length(len(Xy_train)),
    EPOCHS,
    validation_data=generate_data_from_driving(Xy_valid),
    nb_val_samples=3 * epoch_data_length(len(Xy_valid)),
    callbacks=[checkpoint, early_stopping]
)

# evaluate the model
evaluation = model.evaluate_generator(generate_data_from_driving(Xy_test), epoch_data_length(len(Xy_test)))
print("Test evaluation {}".format(evaluation))

# save weights
model.save_weights("./model.h5")

# save model
json_data = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(json_data, json_file)

print("Saved model to disk")
