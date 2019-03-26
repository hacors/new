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
    mnist = tf.keras.datasets.mnist.load_data()
    train_images = tf.expand_dims(tf.cast(mnist[the_type][0], tf.float32), -1)
    train_labels = tf.cast(mnist[the_type][1], tf.int32)
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    batch_train_data = train_data.shuffle(20).batch(batch_size)
    return batch_train_data


if __name__ == '__main__':
    # 定义变量
    train_data_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    train_label_ph = tf.placeholder(tf.int32, [None])
    # 定义网络
    net = mycnn.mycnn([[5, 5, 32], [5, 5, 16]], [[50], [10]], train_data_ph)
    # 定义损失
    onehot_label = tf.one_hot(indices=train_label_ph, depth=10, on_value=1.0, off_value=0.0)
    loss_function = tf.reduce_sum(-onehot_label*tf.log(net.output))
    # loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.output, labels=onehot_label))
    tf.summary.scalar('loss', loss_function)
    # 定义训练方法
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss_function)
    # 获取数据
    train_data = get_batch_data(0, 10)
    iterator = train_data.make_one_shot_iterator()
    one_element = iterator.get_next()
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
                image, label = sess.run(one_element)
                # imashow(image)
                loss, sums, _, result = sess.run([loss_function, merger, train_step, net.output], feed_dict={train_data_ph: image, train_label_ph: label})
                writer.add_summary(sums)
                print(loss, ' ', result)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            print('finaly')

os.system('tensorboard --logdir %s --port 6006' % director)
