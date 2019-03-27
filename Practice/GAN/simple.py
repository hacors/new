import tensorflow as tf

tf.enable_eager_execution()


class simple_net():
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.train_itor = self.get_data(0).make_one_shot_iterator()
        self.test_itor = self.get_data(1).make_one_shot_iterator()

    def get_data(self, data_type, data_size=None, batch_size=100):
        datas = tf.expand_dims(tf.cast(self.mnist[data_type][0][:data_size], tf.float32), -1)
        labels = tf.cast(self.mnist[data_type][1][:data_size], tf.int32)
        merge_data = tf.data.Dataset.from_tensor_slices((datas, labels))
        merge_data = merge_data.shuffle(500).batch(batch_size)
        return merge_data

    def train(self):
        tf.global_variables_initializer()
        try:
            temp_data = self.train_itor.get_next()
            predict_label = self.net_result([[5, 5, 16], [5, 5, 16]], [[10], [10]], temp_data[0])
            true_label = tf.one_hot(temp_data[1], 10, 1.0, 0.0)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict_label, labels=true_label))
            grad=tf.GradientTape().gradient(loss,)
            tf.train.AdamOptimizer(0.001).minimize(loss.value)
        except tf.errors.OutOfRangeError:
            print('finished')

    def net_result(self, conv_list: list, fullconn_list: list, data: tf.Tensor):
        result = data
        for conv_shape in conv_list:
            result = self.conv_layer(conv_shape, result)
        result = tf.layers.flatten(result)
        for fullconn_shape in fullconn_list:
            result = self.fullconn_layer(fullconn_shape, result)
        return result

    def conv_layer(self, shape: list, data: tf.Tensor, name='default'):
        shape.insert(-1, data.shape[-1].value)
        weights, biases = self.get_val(shape, name)
        logits = tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding="SAME")
        logits = tf.add(logits, biases)
        logits = tf.nn.leaky_relu(logits)
        logits = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
        return logits

    def fullconn_layer(self, shape: list, data: tf.Tensor, name='default'):
        shape.insert(-1, data.shape[-1].value)
        weights, biases = self.get_val(shape, name)
        logits = tf.matmul(data, weights)
        logits = tf.add(logits, biases)
        return logits

    def get_val(self, shape, name):
        weights = tf.get_variable(name+'w', shape=shape, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name+'b', shape=[shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        return weights, biases


net = simple_net()
net.train()
'''
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import mycnn

director = 'D:/myfile/data/Tensorflow_learn/logs'


def imashow(imadata):
    plt.ion()
    for data in imadata:
        data = data.reshape(data.shape[:-1]).astype(np.uint8)
        plt.imshow(data, cmap='gray')
        plt.pause(0.5)


def get_batch_data(the_type, batch_size):
    with tf.name_scope('all_data'):
        mnist = tf.keras.datasets.mnist.load_data()
        train_images = tf.expand_dims(tf.cast(mnist[the_type][0], tf.float32), -1)
        train_labels = tf.cast(mnist[the_type][1], tf.int32)
        train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        batch_train_datas, batch_train_labels = train_data.shuffle(500).batch(batch_size)
    return batch_train_datas, batch_train_labels


if __name__ == '__main__':
    # 定义变量
    with tf.name_scope('get_data'):
        train_data = get_batch_data(0, 200)
        iterator = train_data.make_one_shot_iterator()
        datas, labels = iterator.get_next()
        train_data_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
        train_label_ph = tf.placeholder(tf.int32, [None])
        datadict = {train_data_ph: datas, train_label_ph: labels}
    # 定义网络
    net = mycnn.mycnn([[5, 5, 16], [5, 5, 16]], [[10], [10]], train_data_ph)
    # 定义损失
    with tf.name_scope('get_loss'):
        onehot_label = tf.one_hot(indices=train_label_ph, depth=10, on_value=1.0, off_value=0.0)
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net.output, labels=onehot_label))
        tf.summary.scalar('loss', loss_function)
    # 定义训练方法
    with tf.name_scope('train_method'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss_function)
    # 启动对话
    with tf.Session() as sess:
        try:
            shutil.rmtree(director)
        except Exception:
            pass
        os.mkdir(director)
        merger = tf.summary.merge_all()
        writer = tf.summary.FileWriter(director, sess.graph)
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                data_dict = sess.run(datadict)
                merge_data, _, loss = sess.run([merger, train_step, loss_function], feed_dict=data_dict)
                writer.add_summary(merge_data)
        except tf.errors.OutOfRangeError:
            print("done")

os.system('tensorboard --logdir %s --port 6006' % director)
'''
