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
INPUT_DIM = 8
PB_FILE_PATH = 'Competition/Mathe_competition/Signal/other_model'
DATASET_PATH = 'Datasets/huawei_signal/train_processed_set/final_train_cut.csv'


def save_model(the_sess):
    pass


def initialize(input_dim, output_dim, name='default'):
    weights = tf.get_variable(name+'w', shape=[input_dim, output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1, seed=666))
    biases = tf.get_variable(name+'b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
    return weights, biases


def full_connected(input_data, input_dim, output_dim, name):
    weights, biases = initialize(input_dim, output_dim, name)
    output_data = tf.matmul(input_data, weights)
    output_data = tf.add(output_data, biases)
    # output_data = tf.nn.leaky_relu(output_data)  # 有激活函数
    return output_data


def rmse(predicts, labels):
    return tf.sqrt(tf.losses.mean_squared_error(predicts, labels))


def data_iterators(true_data):
    target = true_data.pop('RSRP')
    # true_data = (true_data-true_data.mean())/true_data.std()
    tensor = tf.data.Dataset.from_tensor_slices((true_data.values, target.values))
    tensor = tensor.batch(BATCH_SIZE).shuffle(1000).repeat(10000)
    itor = tensor.make_one_shot_iterator()
    return itor


def cv_iterators(cv_data):
    target = cv_data.pop('RSRP')
    # true_data = (true_data-true_data.mean())/true_data.std()
    tensor = tf.data.Dataset.from_tensor_slices((cv_data.values, target.values))
    tensor = tensor.shuffle(1000).batch(BATCH_SIZE*8)
    itor = tensor.make_one_shot_iterator()
    return itor


if __name__ == '__main__':
    data = pd.read_csv(DATASET_PATH)
    data_itor = data_iterators(data)
    datas = tf.placeholder(tf.float32, shape=[None, INPUT_DIM], name='datas')
    labels = tf.placeholder(tf.float32, shape=[None], name='labels')

    layer1 = full_connected(datas, INPUT_DIM, 20, '1')
    layer2 = tf.nn.tanh(layer1)
    layer3 = full_connected(layer2, 20, 10, '2')
    layer4 = tf.nn.tanh(layer3)
    layer5 = full_connected(layer4, 10, 10, '3')
    layer6 = full_connected(layer5, 10, 10, '4')

    # connc_2 = full_connected(connc_1, 10, 1, '2')

    predicts = tf.reduce_mean(layer6, 1)  # 去除最后一维

    loss = rmse(predicts, labels)
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=tf.trainable_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        index = 0
        try:
            while True:
                index += 1
                next_batch_datas, next_batch_labels = data_itor.get_next()
                real_batch_datas, real_batch_labels = sess.run([next_batch_datas, next_batch_labels])
                feed_dict = {datas: real_batch_datas, labels: real_batch_labels}
                blank, the_loss, the_pridict = sess.run([train_step, loss, predicts], feed_dict=feed_dict)
                if index % 10000 == 0:
                    print(the_loss)
                    new_dir = PB_FILE_PATH + '/temp_model_%s' % index
                    if os.path.exists(new_dir):
                        shutil.rmtree(new_dir)
                    else:
                        os.mkdir(new_dir)
                    tf.saved_model.simple_save(sess, new_dir, {"myInput": datas}, outputs={"myOutput": predicts})
        except tf.errors.OUT_OF_RANGE:
            pass
        finally:
            pass
