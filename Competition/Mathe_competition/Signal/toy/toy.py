import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.saved_model import tag_constants

import process_data

BATCH_SIZE = 32
# input_dim = 17
PB_FILE_PATH = r'D:\code\Competition\Mathe_competition\Signal\toy\model_save'
dataset_path = r'D:\code\Datasets\huawei_signal\train_processed_set\train.csv'
# dataset_path = '/root/code/Datasets/huawei_signal/train_processed_set/train.csv'
variable_path = os.path.join(PB_FILE_PATH, 'variables')


def save_model():
    pass


def load_model(path):
    saver = tf.train.Saver()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], PB_FILE_PATH)
        saver.restore(sess, variable_path)
        pass


def initialize(input_dim, output_dim, name='default'):
    weights = tf.get_variable(name+'w', shape=[input_dim, output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1, seed=666))
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
    tensor = tensor.shuffle(1000).batch(BATCH_SIZE).repeat(100)
    itor = tensor.make_one_shot_iterator()
    return itor


if __name__ == '__main__':
    toy_data = process_data.my_process(dataset_path).pop('RSRP')
    toy_data = toy_data[:100]
    data_tensor = tf.data.Dataset.from_tensor_slices((toy_data)).batch(10)
    data_itor = data_tensor.make_one_shot_iterator()

    model_input = tf.placeholder(tf.float32, shape=[None], name='datas')
    labels = tf.placeholder(tf.float32, shape=[None], name='labels')
    predicts = model_input
    varias = tf.get_variable('default', shape=[1], dtype=tf.float32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        temp_tensor = data_itor.get_next()
        true_data = sess.run(temp_tensor)
        model_output = sess.run(predicts, feed_dict={model_input: true_data})
        tf.saved_model.simple_save(sess, PB_FILE_PATH, {"myInput": model_input}, outputs={"myOutput": predicts})
'''
import tensorflow as tf

if __name__=='__main__':

    n = 17

    X = tf.placeholder(tf.float32, shape=[None,n], name='input')  # input
    W1 = tf.get_variable("W1", [n,1], initializer=tf.zeros_initializer())
    b1 = tf.constant(65.245615,tf.float32)
    Z = tf.matmul(X,W1) + b1     # outputs

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(Z,feed_dict={X:})

        # 保存模型
        tf.saved_model.simple_save(sess, './model/', inputs={"myInput": X}, outputs={"myOutput": Z})
'''