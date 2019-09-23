import os

import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
from sklearn import preprocessing
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.saved_model import tag_constants
import sklearn
import process_data
BATCH_SIZE = 128
FILE_SIZE = 133
INPUT_DIM = 8
PB_FILE_PATH = 'Competition/Mathe_competition/Signal/save_model'
DATASET_PATH = 'Datasets/huawei_signal/train_processed_set/cuted_data'


def save_model(the_sess):
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
    tensor = tensor.batch(BATCH_SIZE)
    itor = tf.data.Iterator.from_structure(tensor.output_types, tensor.output_shapes)
    return itor


def cv_iterators(cv_data):
    target = cv_data.pop('RSRP')
    # true_data = (true_data-true_data.mean())/true_data.std()
    tensor = tf.data.Dataset.from_tensor_slices((cv_data.values, target.values))
    tensor = tensor.shuffle(1000).batch(BATCH_SIZE*8)
    itor = tensor.make_one_shot_iterator()
    return itor


if __name__ == '__main__':
    path_list = os.listdir(DATASET_PATH)
    train_list = path_list[:1000]
    cv_list = path_list[1000:]

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
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list=tf.trainable_variables())
    init = tf.global_variables_initializer()

    first_file_data = pd.read_csv(os.path.join(DATASET_PATH, str(0)+'.csv'))
    first_file_target = first_file_data.pop('RSRP')
    # true_data = (true_data-true_data.mean())/true_data.std()
    first_file_tensor = tf.data.Dataset.from_tensor_slices((first_file_data.values, first_file_target.values)).batch(BATCH_SIZE)
    iterator = tf.data.Iterator.from_structure(first_file_tensor.output_types, first_file_tensor.output_shapes)

    with tf.Session() as sess:
        sess.run(init)
        for repeat in range(10):
            for file_index in range(1, 1000):
                temp_file_data = pd.read_csv(os.path.join(DATASET_PATH, str(file_index)+'.csv'))
                temp_file_target = temp_file_data.pop('RSRP')
                temp_file_tensor = tf.data.Dataset.from_tensor_slices((temp_file_data.values, temp_file_target.values)).batch(BATCH_SIZE)
                sess.run(iterator.make_initializer(temp_file_tensor))
                try:
                    while True:
                        next_batch_datas, next_batch_labels = iterator.get_next()
                        real_batch_datas, real_batch_labels = sess.run([next_batch_datas, next_batch_labels])
                        feed_dict = {datas: real_batch_datas, labels: real_batch_labels}
                        blank, the_loss, the_predict = sess.run([train_step, loss, predicts], feed_dict=feed_dict)
                except Exception:
                    print(the_loss)
                    '''
                    temp_index = np.random.randint(900, 1065)
                    temp_cv_file = pd.read_csv(os.path.join(DATASET_PATH, str(temp_index)+'.csv'))
                    cv_itor = cv_iterators(temp_cv_file)
                    cv_datas, cv_labels = cv_itor.get_next()
                    real_cv_data, real_cv_labels = sess.run([cv_datas, cv_labels])
                    cv_loss = sess.run(loss, feed_dict={datas: real_cv_data, labels: real_cv_labels})
                    print(cv_loss)
                    '''
                finally:
                    pass
                    # tf.delete_session_tensor(data_itor)
                    # 需要释放产生器
                if file_index % 50 == 0:
                    new_dir = PB_FILE_PATH + '/temp_model_%s_%s' % (repeat, file_index)
                    if os.path.exists(new_dir):
                        shutil.rmtree(new_dir)
                    else:
                        os.mkdir(new_dir)
                    tf.saved_model.simple_save(sess, new_dir, {"myInput": datas}, outputs={"myOutput": predicts})
