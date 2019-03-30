# -*- coding: utf-8 -*-
import argparse
import os
import time

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)
mnist = tf.keras.datasets.mnist.load_data()
train_data, train_label, test_data, test_label = mnist[0][0], mnist[0][1], mnist[1][0], mnist[1][1]


def discrimanator(input_shape, conv_list: list, dens_list: list):
    with tf.variable_scope('discriminator'):
        input_data = keras.layers.Input(shape=input_shape)
        digits = input_data
        for dim in conv_list:
            digits = keras.layers.Conv3D(dim, (5, 5, 1), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
            digits = keras.layers.MaxPool2D()(digits)
        digits = keras.layers.Flatten()(digits)
        for dim in dens_list:
            digits = keras.layers.Dense(dim, activation=keras.layers.LeakyReLU, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        prediction = keras.layers.Softmax()
        models = keras.Model(inputs=input_data, outputs=prediction)
        return model


mydis = discrimanator((28, 28, 1), [12, 12], [10, 10])
