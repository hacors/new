import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

import dataset
tf.enable_eager_execution()
BATCHSIZE = 1
KL = keras.layers
init = keras.initializers.RandomNormal(stddev=0.01)
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
    pic_true = tf.cast(pic_true, tf.float32)
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


def spatial_net():
    input_data = keras.Input(shape=(160, 160, 3))
    digits = input_data
    digits = KL.Conv2D(20, (5, 5), activation='relu',  padding='same')(digits)
    digits = KL.MaxPool2D((2, 2), padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.MaxPool2D((2, 2), padding='same')(digits)

    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(40, (3, 3), activation='relu',  padding='same')(digits)

    digits = KL.Conv2D(20, (3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv2D(10, (3, 3), activation='relu',  padding='same')(digits)

    digits = KL.Conv2D(1, (1, 1), activation='relu',  padding='same')(digits)

    prediction = digits
    spatial_net = keras.Model(inputs=input_data, outputs=prediction)
    return spatial_net


def temporal_net():
    input_data = keras.Input(shape=(5, 160, 160, 3))
    digits = input_data
    digits = KL.Conv3D(20, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(40, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(20, (3, 3, 3), activation='relu',  padding='same')(digits)
    digits = KL.Conv3D(1, (1, 1, 1), activation='relu',  padding='same')(digits)
    prediction = digits
    temporal_net = keras.Model(inputs=input_data, outputs=prediction)
    return temporal_net
def fusion_net():
    input_data=
    t_net = temporal_net()

if __name__ == "__main__":
    tfrecord_path = os.path.join(dataset.set_root, dataset.set_name, 'train.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(BATCHSIZE)
    temp_net = temporal_net()
    for epoch in range(10):
        for index, data in enumerate(batched_dataset):
            # for repeat in range(20):
            with tf.GradientTape() as train_tape:
                opti = tf.train.AdamOptimizer(learning_rate=1e-5)
                predict = temp_net(data[0], training=True)  # 注意所有的keras模型必须添上一句话，training=True
                '''
                loss = euclidean_distance_loss(data[1], predict)
                gradiens = train_tape.gradient(loss, mynet.variables)
                opti.apply_gradients(zip(gradiens, mynet.variables))
                '''
