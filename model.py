import numpy as np
import math
import csv
import json

from scipy.stats import bernoulli

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

EPOCHS = 20
BATCH = 100
CHANNEL_NUMBER = 3
ANGLE_COEFFICIENT = 0.229
FINE_TUNING = False
FAIR_COIN = 0.5

CROPPING = (0, 0, 0, 0)

CHOICE_LIST = ['CENTER', 'LEFT', 'RIGHT']

SHAPE = (preprocessing.POST_PROCESSING_SIZE, preprocessing.POST_PROCESSING_SIZE, CHANNEL_NUMBER)

tf.python.control_flow_ops = tf


def learning_rate():
    if FINE_TUNING:
        return 0.000001
    else:
        return 0.0001


def generate_cnn_model():
    cnn = Sequential()
    cnn.add(Lambda(lambda x: x / 127.5 - 1., input_shape=SHAPE, output_shape=SHAPE))
    cnn.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=SHAPE))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Convolution2D(64, 3, 3))
    cnn.add(Dropout(0.5))
    cnn.add(LeakyReLU())
    cnn.add(MaxPooling2D((2, 2), (1, 1)))
    cnn.add(Flatten())
    cnn.add(Dense(100))
    cnn.add(LeakyReLU())
    cnn.add(Dense(50))
    cnn.add(LeakyReLU())
    cnn.add(Dense(10))
    cnn.add(LeakyReLU())
    cnn.add(Dense(1))
    cnn.summary()
    return cnn


def generate_data_from_driving(images):
    closed_counter = 0
    while 1:
        x = np.ndarray((BATCH, *SHAPE), dtype=float)
        y = np.ndarray(BATCH, dtype=float)
        for k in range(BATCH):
            if closed_counter == len(images):
                shuffle(images)
                closed_counter = 0

            random_image = np.random.choice(CHOICE_LIST)
            mirrored = bernoulli.rvs(FAIR_COIN)

            if random_image == CHOICE_LIST[0]:
                filename = images[closed_counter][0]
                angle = images[closed_counter][3]
                if mirrored:
                    angle = -angle
            elif random_image == CHOICE_LIST[1]:
                filename = images[closed_counter][1]
                angle = images[closed_counter][3] + ANGLE_COEFFICIENT
                if mirrored:
                    angle = -angle
            else:
                filename = images[closed_counter][2]
                angle = images[closed_counter][3] - ANGLE_COEFFICIENT
                if mirrored:
                    angle = -angle

            x[k] = preprocessing.pre_processing(filename, mirrored, angle)
            y[k] = angle
            closed_counter += 1
        yield ({'lambda_input_1': x}, {'dense_4': y})


def epoch_data_length(data_length):
    return math.ceil(data_length / BATCH) * BATCH


with open(preprocessing.SELECTED_FOLDER + 'driving_log.csv', 'r') as file:
    reader = csv.reader(file)
    driving_log = list(reader)

driving_log_length = len(driving_log)

X_train = [("", "", "", 0.0) for x in range(driving_log_length)]

for i in range(driving_log_length):
    X_train[i] = (
        driving_log[i][0].lstrip(),
        driving_log[i][1].lstrip(),
        driving_log[i][2].lstrip(),
        float(driving_log[i][3])
    )

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

model.compile(Adam(learning_rate), 'mse')

checkpoint = ModelCheckpoint('model.h5', 'val_loss', 1, True)
early_stopping = EarlyStopping('val_loss', patience=5, verbose=1)

history = model.fit_generator(
    generate_data_from_driving(X_train),
    epoch_data_length(len(X_train)),
    EPOCHS,
    validation_data=generate_data_from_driving(X_valid),
    nb_val_samples=epoch_data_length(len(X_valid)),
    callbacks=[checkpoint, early_stopping]
)

evaluation = model.evaluate_generator(generate_data_from_driving(X_test), epoch_data_length(len(X_test)))

print("Test evaluation {}".format(evaluation))

model.save_weights("./model.h5")

json_data = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(json_data, json_file)

print("Saved model to disk")
