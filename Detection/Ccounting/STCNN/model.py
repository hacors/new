import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

import dataset
tf.enable_eager_execution()
BATCHSIZE = 1
KL = keras.layers
feature = {
    'pic': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string),
}


def parse_image_function(example_proto):  # 解码
    return tf.parse_single_example(example_proto, feature)


def process_function(parsed_data):
    pic_string = parsed_data['pic']
    dens_string = parsed_data['dens']
    pic_true = tf.reshape(tf.decode_raw(pic_string, tf.uint8), [dataset.video_length, dataset.crop_size, dataset.crop_size, 3])
    dens_true = tf.reshape(tf.decode_raw(dens_string, tf.float32), [dataset.video_length, dataset.crop_size, dataset.crop_size, 1])  # 注意图片必须是三维的
    '''
    for index in range(5):
        temp_pic = pic_true.numpy()
        temp_dens = np.squeeze(dens_true.numpy())
        plt.imshow(temp_pic[index])
        plt.show()
        plt.imshow(temp_dens[index])
        plt.show()
    '''
    return pic_true, dens_true


def spatial_string():
    input_data = keras.Input(shape=(160, 160, 3))
    digits = input_data 
    digits = KL.Conv2D(20, (5, 5), activation='relu', kernel_initializer=init, padding='same')(digits)
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
    tfrecord_path = os.path.join(dataset.set_root, dataset.set_name, 'train.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    for temp in parsed_dataset:
        process_function(temp)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(BATCHSIZE)
