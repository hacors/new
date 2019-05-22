import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import os

import process

tf.enable_eager_execution()
KL = keras.layers
VGG19 = keras.applications.vgg19.VGG19
feature = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'img': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string),
}


def parse_image_function(example_proto):  # 解码
    return tf.parse_single_example(example_proto, feature)


def process_function(parsed_data):
    height = parsed_data['height']
    width = parsed_data['width']
    img_string = parsed_data['img']
    dens_string = parsed_data['dens']
    img_true = tf.reshape(tf.decode_raw(img_string, tf.uint8), [height, width, 3])
    dens_true = tf.reshape(tf.decode_raw(dens_string, tf.uint8), [height, width, 1])  # 注意图片必须是三维的
    img_casted = tf.cast(img_true, tf.float32)
    dens_casted = tf.cast(dens_true, tf.float32)
    img_processed = tf.divide(img_casted, 255.0)
    dens_processed = tf.image.resize_images(dens_casted, [height/8, width/8], method=2)
    dens_processed = tf.divide(dens_processed, 255.0)
    return img_processed, dens_processed


def euclidean_distance_loss(y_true, y_pred):
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    return loss


def crowd_net():
    init = keras.initializers.RandomNormal(stddev=0.01)
    vgg_tune = VGG19(weights='imagenet', include_top=False)
    input_data = keras.Input(shape=(None, None, 3))
    digits = vgg_tune(input_data)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=init, padding='same')(digits)
    digits = KL.Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init, padding='same')(digits)
    prediction = digits
    crowd_net = keras.Model(inputs=input_data, outputs=prediction)
    return crowd_net


if __name__ == "__main__":
    shtech_image_path, shtech_set_path = process.get_shtech_path()
    tfrecord_path = os.path.join(shtech_set_path[0][0], 'all_data.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(9).repeat(10)  # 每个batch都是同一张图片切出来的
    mynet = crowd_net()
    for dataset in batched_dataset:
        train_tape = tf.GradientTape()
        opti = tf.train.GradientDescentOptimizer()
        predict = mynet(dataset[0])
        loss = euclidean_distance_loss(dataset[1], predict)
        gradiens = train_tape.gradient(loss, mynet.variables)
        opti.apply_gradients(zip(gradiens, mynet.variables))
