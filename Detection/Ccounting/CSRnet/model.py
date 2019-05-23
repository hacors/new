import tensorflow as tf
from tensorflow import keras
import os

import process

tf.enable_eager_execution()
KL = keras.layers
VGG16 = keras.applications.vgg16.VGG16
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
    img_expand = tf.expand_dims(img_processed, -1)
    img_part_0 = tf.divide(tf.subtract(img_expand[:, :, 0, :], 0.485), 0.229)
    img_part_1 = tf.divide(tf.subtract(img_expand[:, :, 1, :], 0.456), 0.224)
    img_part_2 = tf.divide(tf.subtract(img_expand[:, :, 2, :], 0.406), 0.225)
    img_merged = tf.concat([img_part_0, img_part_1, img_part_2], 2)
    dens_processed = tf.image.resize_images(dens_casted, [height/8, width/8], method=2)
    # dens_processed = tf.divide(dens_processed, 255.0)
    return img_merged, dens_processed


def euclidean_distance_loss(y_true, y_pred):
    loss_metrix = keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_pred - y_true), axis=-1))
    return loss_metrix


def crowd_net():
    init = keras.initializers.RandomNormal(stddev=0.01)
    vgg = VGG16(weights='imagenet', include_top=False)  # W
    input_data = keras.Input(shape=(None, None, 3))
    crop_vgg = keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output)  # 注意这是模型截取的写法
    digits = crop_vgg(input_data)
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
        opti = tf.train.GradientDescentOptimizer(learning_rate=0.03)
        predict = mynet(dataset[0])
        loss = euclidean_distance_loss(dataset[1], predict)
        gradiens = train_tape.gradient(loss, mynet.variables)
        opti.apply_gradients(zip(gradiens, mynet.variables))
