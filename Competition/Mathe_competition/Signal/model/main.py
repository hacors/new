import os

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.saved_model import tag_constants

import process_data

BATCH_SIZE = 32
# input_dim = 17
pb_file_path = r'D:\code\Competition\Mathe_competition\Signal\model'
dataset_path = r'D:\code\Datasets\huawei_signal\train_processed_set\train.csv'
variable_path = os.path.join(pb_file_path, 'variables')


def save_model():
    pass


def load_model(path):
    saver = tf.train.Saver()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_file_path)
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
    output_data = tf.nn.leaky_relu(output_data)  # 有激活函数
    return output_data


def rmse(predicts, labels):
    return  tf.sqrt(tf.losses.mean_squared_error(predicts, labels))


def data_iterators(true_data):
    target = true_data.pop('RSRP')
    # true_data = (true_data-true_data.mean())/true_data.std()

    tensor = tf.data.Dataset.from_tensor_slices((true_data.values, target.values))
    tensor = tensor.shuffle(500).batch(BATCH_SIZE).repeat(1)
    itor = tensor.make_one_shot_iterator()
    return itor


if __name__ == '__main__':
    true_data = process_data.my_process(dataset_path)
    input_dim = true_data.shape[-1]-1
    data_itor = data_iterators(true_data)
    datas = tf.placeholder(tf.float32, shape=[None, input_dim], name='datas')
    labels = tf.placeholder(tf.float32, shape=[None], name='labels')

    connc_1 = full_connected(datas, input_dim, 10, '1')
    connc_2 = full_connected(connc_1, 10, 1, '2')
    predicts = tf.reduce_mean(connc_2, 1)  # 去除最后一维
    loss = rmse(predicts, labels)

    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, var_list=tf.trainable_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([init])
        for i in range(100):
            next_batch_datas, next_batch_labels = data_itor.get_next()
            real_batch_datas, real_batch_labels = sess.run([next_batch_datas, next_batch_labels])
            feed_dict = {datas: real_batch_datas, labels: real_batch_labels}
            _, the_loss = sess.run([train_step, loss], feed_dict=feed_dict)
            print(the_loss)
        '''
        try:
            while True:
                next_batch_datas, next_batch_labels = data_itor.get_next()
                real_batch_datas, real_batch_labels = sess.run([next_batch_datas, next_batch_labels])
                feed_dict = {datas: real_batch_datas, labels: real_batch_labels}
                result = sess.run([train_step], feed_dict=feed_dict)
                the_loss = sess.run([loss])
        except Exception:
            print('end')
        finally:
            pass
        '''
