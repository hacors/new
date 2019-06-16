import tensorflow as tf
import dataset
import os
from matplotlib import pyplot as plt
import numpy as np
tf.enable_eager_execution()
feature = {
    'pic': tf.FixedLenFeature([], tf.string),
    'dens': tf.FixedLenFeature([], tf.string),
}
BATCHSIZE = 1


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


if __name__ == "__main__":
    tfrecord_path = os.path.join(dataset.set_root, dataset.set_name, 'train.tfrecords')
    tfrecord_file = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = tfrecord_file.map(parse_image_function)
    for temp in parsed_dataset:
        process_function(temp)
    processed_dataset = parsed_dataset.map(process_function)
    batched_dataset = processed_dataset.batch(BATCHSIZE)
