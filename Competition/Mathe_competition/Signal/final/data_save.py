import os

import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
from sklearn import preprocessing
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.saved_model import tag_constants

import customize_service
BATCH_SIZE = 128
FILE_SIZE = 133
INPUT_DIM = 10
PB_FILE_PATH = 'Competition/Mathe_competition/Signal/other_model'
DATASET_PATH = 'Datasets/huawei_signal/train_processed_set/train_merge.csv'
TF_PATH = 'Datasets/huawei_signal/train_processed_set/tf_cord/train.tfrecords'
train_frame = customize_service.get_from_path(DATASET_PATH)
train_labels_frame = train_frame.pop('RSRP')

train_values = train_frame.values
train_size = train_values.shape[0]
train_labels_values = train_labels_frame.values

# print(train_values[0].shape)
# print(train_values[0].dtype)
# print(train_labels_values[0])

# ------------------------------create TFRecord file----------------------------#
writer = tf.python_io.TFRecordWriter(path=TF_PATH)
for i in range(train_size):
    datas = train_values[i].tostring()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "datas": tf.train.Feature(bytes_list=tf.train.BytesList(value=[datas])),
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[train_labels_values[i]]))
            }
        )
    )
    writer.write(record=example.SerializeToString())
writer.close()
