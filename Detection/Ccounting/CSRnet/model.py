import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
import os

import process

tf.enable_eager_execution()
KL = keras.layers
feature = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'img': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string)
}


def parse_image_function(example_proto):
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
    init_func = keras.initializers.RandomNormal(stddev=0.01)
    k_size = (3, 3)

    model = keras.Sequential()
    model.add(KL.Conv2D(64, kernel_size=k_size, input_shape=(None, None, 3), activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(64, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPooling2D(strides=2))
    model.add(KL.Conv2D(128, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(128, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPooling2D(strides=2))
    model.add(KL.Conv2D(256, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(256, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(256, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.MaxPooling2D(strides=2))
    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())
    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', padding='same'))
    model.add(KL.BatchNormalization())

    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))
    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))
    model.add(KL.Conv2D(512, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))
    model.add(KL.Conv2D(256, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))
    model.add(KL.Conv2D(128, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))
    model.add(KL.Conv2D(64, kernel_size=k_size, activation='relu', dilation_rate=2, kernel_initializer=init_func, padding='same'))

    model.add(KL.Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=init_func, padding='same'))

    opti = keras.optimizers.SGD(lr=1e-7, decay=(5*1e-4), momentum=0.95)
    loss
    model.compile()


if __name__ == "__main__":
    shtech_image_path, shtech_set_path = process.get_shtech_path()
    tfrecord_path = os.path.join(shtech_set_path[0][0], 'all_data.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.shuffle(1000).batch(10).repeat(10)
    temp = batched_dataset.take(1)
    pass
