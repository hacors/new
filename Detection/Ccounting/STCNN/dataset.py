'''
获取对应数据集的tfcord文件
'''
import glob  # 获取文件名列表
import os
import random

import h5py
import numpy as np
import tensorflow as tf

video_length = 5
crop_size = 160
crop_num = 8
set_name = 'mall'
set_root = 'Datasets'


def organize_pathdata(processed_path):  # 将所有的图片组织起来
    file_list = glob.glob(os.path.join(processed_path, '*.h5'))
    data_info = dict()
    for file_path in file_list:  # 按视频分组
        video_index = int(file_path.split('_')[-4])
        if data_info.get(video_index):
            data_info[video_index].append(file_path)
        else:
            data_info[video_index] = [file_path]
    data_set = list()
    for video in data_info:
        pic_list = data_info.get(video)
        for index in range(len(pic_list)):
            if index+video_length <= len(pic_list):
                group_pic = pic_list[index:index+video_length]
                data_set.append(group_pic)
    return data_set


def crop_function(pic_array, dens_array):  # 随机截取图片，获取训练数据集
    pic_list = list()
    dens_list = list()
    shape = pic_array.shape[1:3]
    assert shape[0] >= crop_size and shape[1] >= crop_size
    for repeat in range(crop_num):
        crop_of_0 = random.randint(0, shape[0]-crop_size)
        crop_of_1 = random.randint(0, shape[1]-crop_size)
        crop_of_pic = pic_array[:, crop_of_0:crop_of_0+crop_size, crop_of_1:crop_of_1+crop_size, :]
        crop_of_dens = dens_array[:, crop_of_0:crop_of_0+crop_size, crop_of_1:crop_of_1+crop_size, :]
        pic_list.append(crop_of_pic)
        dens_list.append(crop_of_dens)
    return pic_list, dens_list


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tfrecord(struct_pathdata, tfrecord_path):  # 将所有的连续帧变成tfcord文件
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for video_paths in struct_pathdata:
        sum_pic = list()
        sum_dens = list()
        for file_path in video_paths:
            h5_file = h5py.File(file_path, 'r')
            pic_file = h5_file['pic'][()]
            dens_file = h5_file['dens'][()]
            sum_pic.append(pic_file)
            sum_dens.append(dens_file)
        pic_array = np.array(sum_pic)
        dens_array = np.array(sum_dens)
        dens_array = np.expand_dims(dens_array, -1).astype(np.float32)
        pic_crop_list, dens_crop_list = crop_function(pic_array, dens_array)
        for index in range(crop_num):
            temp_raw_pic = pic_crop_list[index].tostring()
            temp_raw_dens = dens_crop_list[index].tostring()
            feature = {
                'pic': bytes_feature(temp_raw_pic),
                'dens': bytes_feature(temp_raw_dens)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        # print(video_paths)
    writer.close()
    print('down')


if __name__ == "__main__":
    struct_pathdata = organize_pathdata(os.path.join(set_root, set_name, 'train_processed'))
    tfrecord_path = os.path.join(set_root, set_name, 'train.tfrecords')
    get_tfrecord(struct_pathdata, tfrecord_path)
