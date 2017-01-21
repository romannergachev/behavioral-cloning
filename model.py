import numpy as np
import math
import csv
import cv2
import json
import util

from keras.applications import ResNet50
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Input
from keras.engine import Model
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
import tensorflow as tf

EPOCHS = 20
BATCH = 100
WIDTH = 160
# WiDTH = 100
# HEIGHTS = 50
HEIGHTS = 80
CHANNEL_NUMBER = 3
REG = 0.0
FINE_TUNING = False

CROPPING = (0, 0, 0, 0)

SHAPE = (HEIGHTS, WIDTH, CHANNEL_NUMBER)

INITIAL_ITERATION = 'data/'
FIRST_ITERATION = 'driving/'
SECOND_ITERATION = 'driving2/'
THIRD_ITERATION = 'not/track1_recovery/'
ERRORS_ITERATION = 'errors/'

SELECTED_FOLDER = INITIAL_ITERATION

tf.python.control_flow_ops = tf


def learning_rate():
    if FINE_TUNING:
        return 0.0001
    else:
        return 0.0001


def crop_image(img, cropping):
    return img[cropping[0]:img.shape[0] - cropping[1], cropping[2]:img.shape[1] - cropping[3], :]


def get_cropped_shape(img_shape, cropping):
    return (img_shape[0] - cropping[0] - cropping[1],
            img_shape[1] - cropping[2] - cropping[3],
            img_shape[2])


def apply_clahe(filename, mirrored=False):
    image = cv2.imread(INITIAL_ITERATION + filename)
    # image = cv2.imread(filename)
    image = cv2.resize(image, (WIDTH, HEIGHTS))
    image = crop_image(image, CROPPING)
    if mirrored:
        image = cv2.flip(image, 1)

    img = image[np.newaxis, ...]
    return img


def generate_cnn_model():
    cnn = Sequential()
    cnn.add(Lambda(lambda x: x / 127.5 - 1., input_shape=get_cropped_shape(SHAPE, CROPPING), output_shape=SHAPE))
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


def vgg16_model():
    input_layer = Input(shape=get_cropped_shape(SHAPE, CROPPING))
    base_model = VGG16(include_top=False, input_tensor=input_layer)

    # Remove the last block of the VGG16 net.
    [base_model.layers.pop() for _ in range(4)]
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    # Make sure pre trained layers from the VGG net don't change while training.
    for layer in base_model.layers:
        layer.trainable = False

    # Add last block to the VGG model with modified sub sampling.
    layer = base_model.outputs[0]
    layer = Convolution2D(512, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='block5_conv1')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv2')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv3')(
        layer)
    layer = Dropout(.5)(layer)
    layer = Flatten()(layer)
    layer = Dense(2048, activation='relu', name='fc1')(layer)
    layer = Dense(1024, activation='relu', name='fc2')(layer)
    layer = Dropout(.5)(layer)
    layer = Dense(1, activation='linear', name='predictions')(layer)

    return Model(input=base_model.input, output=layer)


def generate_data_from_driving(images):
    closed_counter = 0
    while 1:
        x = np.ndarray((BATCH, *get_cropped_shape(SHAPE, CROPPING)), dtype=float)
        y = np.ndarray(BATCH, dtype=float)
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
            x[k] = final_image
            y[k] = angle
            closed_counter += 1
        yield ({'lambda_input_1': x}, {'dense_4': y})


def epoch_data_length(data_length):
    return math.ceil(data_length / BATCH) * BATCH


with open(SELECTED_FOLDER + 'driving_log.csv', 'r') as file:
    reader = csv.reader(file)
    driving_log = list(reader)

driving_log_length = len(driving_log)

X_train = [("", 0.0, False) for x in range(driving_log_length)]

for i in range(driving_log_length):
    X_train[i] = (driving_log[i][0].lstrip(), float(driving_log[i][3]), 0)

driving_log_length = len(X_train)

if not False:
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

model.compile(Adam(learning_rate), 'mse')

checkpoint = ModelCheckpoint('model.h5', 'val_loss', 1, True)
early_stopping = EarlyStopping('val_loss', patience=5, verbose=1)

train_gen = util.generate_next_batch()
validation_gen = util.generate_next_batch()

history = model.fit_generator(
    generate_data_from_driving(X_train),
    epoch_data_length(len(X_train)),
    EPOCHS,
    validation_data=generate_data_from_driving(X_valid),
    nb_val_samples=epoch_data_length(len(X_valid)),
    callbacks=[checkpoint, early_stopping]
)

# ImageDataGenerator()

accuracy = model.evaluate_generator(generate_data_from_driving(X_test), epoch_data_length(len(X_test)))

print("Test accuracy {}".format(accuracy))

model.save_weights("./model.h5")

json_data = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(json_data, json_file)

print("Saved model to disk")
