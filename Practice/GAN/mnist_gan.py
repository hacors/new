# -*- coding: utf-8 -*-
import argparse
import os
import time

import tensorflow as tf

tf.enable_eager_execution()

KL=tf.keras.layers
def discrimanator(input_shape):
    with tf.variable_scope('discriminator'):
        lay=KL.
'''
def create_model():
    
    kL = tf.keras.layers
    maxpool = KL.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", data_format=data_format)
    return tf.keras.Sequential([
        KL.Reshape(target_shape=input_shape, input_shape=(28*28, )),
        KL.Conv2D(filters=32, kernel_size=5, padding="same", data_format=data_format, activation=tf.nn.relu),
        maxpool,
        KL.Conv2D(filters=64, kernel_size=5, padding="same", data_format=data_format, activation=tf.nn.relu),
        maxpool,
        KL.Flatten(),  # 卷积结果压成一维
        KL.Dense(units=1024, activation=tf.nn.relu),
        KL.Dropout(rate=0.5),
        KL.Dense(units=10)
    ])


def loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def compute_accuracy(logits, labels):
    predictions = tf.argmax(input=logits, axis=1, output_type=tf.int64)
    labels = tf.cast(x=labels, dtype=tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(model, optimizer, dataset, step_counter, log_interval=None):
    """在数据集上训练模型"""
    start = time.time()
    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(logits, labels)
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)
        if log_interval and batch % log_interval == 0:
            rate = log_interval / (time.time() - start)
            print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
            start = time.time()


def test(model, dataset):
    """模型会在测试数据集上进行评估"""
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    for (images, labels) in tfe.Iterator(dataset):
        logits = model(images, training=False)
        avg_loss(loss(logits, labels))
        accuracy(tf.argmax(logits, axis=1, output_type=tf.int64), tf.cast(labels, tf.int64))
    print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
          (avg_loss.result(), 100 * accuracy.result()))


def run_mnist_eager(cfg):
    # 自动选择设备
    (device, data_format) = ('/gpu:0', 'channels_first')
    if not tf.test.is_gpu_available():
        (device, data_format) = ('/cpu:0', 'channels_last')

    print('Using device %s, and data format %s.' % (device, data_format))
    # 载入数据集
    # 方式1
    # train_ds = mnist_dataset.train(cfg.data_dir).shuffle(60000, reshuffle_each_iteration=True).batch(cfg.batch_size)
    # test_ds = mnist_dataset.test(cfg.data_dir).batch(cfg.batch_size)
    # 方式2
    train_ds, test_ds = load_mnist()  # shape = (?, 768) / (?)
    train_ds = train_ds.shuffle(60000, reshuffle_each_iteration=True).batch(cfg.batch_size)
    test_ds = test_ds.batch(cfg.batch_size)
    # print(train_ds.output_shapes, test_ds.output_shapes)

    # 创建 model and optimizer
    model = create_model(data_format=data_format)
    optimizer = tf.train.MomentumOptimizer(cfg.lr, cfg.momentum)

    # Create and restore checkpoint (if one exists on the path)
    checkpoint_prefix = os.path.join(cfg.model_dir, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tfe.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)
    # 从检查点文件恢复模型参数, 如果文件存在.
    checkpoint.restore(tf.train.latest_checkpoint(cfg.model_dir))

    # Train and evaluate for a set number of epochs.
    with tf.device(device):  # 使用GPU必须有此一句
        for _ in range(cfg.train_epochs):
            start = time.time()
            train(model, optimizer, train_ds, step_counter, cfg.log_interval)
            end = time.time()
            print('\nTrain time for epoch #%d (%d total steps): %f' %
                  (checkpoint.save_counter.numpy() + 1, step_counter.numpy(), end - start))

            test(model, test_ds)
            checkpoint.save(checkpoint_prefix)


def arg_parse():
    """参数定义"""
    parser = argparse.ArgumentParser(description="Lenet-5 MNIST 模型")
    parser.add_argument("--lr", dest="lr", help="学习率", default=0.01, type=float)
    parser.add_argument("--momentum", dest="momentum", help="SGD momentum.", default=0.5)

    parser.add_argument("--data_dir", dest="data_dir", help="数据集下载/保存目录", default="data/mnist/input_data/")
    parser.add_argument("--model_dir", dest="model_dir", help="模型保存目录", default="data/mnist/checkpoints/")
    parser.add_argument("--batch_size", dest="batch_size", help="训练或测试时 Batch Size", default=100, type=int)
    parser.add_argument("--train_epochs", dest="train_epochs", help="训练时epoch迭代次数", default=4, type=int)
    parser.add_argument("--log_interval", dest="log_interval", help="日志打印间隔", default=10, type=int)

    # 返回转换好的结果
    return parser.parse_args()


def load_mnist():

    mnist = input_data.read_data_sets(train_dir="data/mnist/input_data", one_hot=False,
                                      source_url="http://yann.lecun.com/exdb/mnist/")
    train = mnist.train
    val = mnist.validation
    # train_ds = tf.data.Dataset.from_tensor_slices({
    #     "images": train.images/255.,
    #     "labels": train.labels
    # })
    # test_ds = tf.data.Dataset.from_tensor_slices({
    #     "images": val.images/255.,
    #     "labels": val.labels
    # })
    train_ds = tf.data.Dataset.from_tensor_slices((
        train.images,
        train.labels.astype(int)))
    test_ds = tf.data.Dataset.from_tensor_slices((
        val.images,
        val.labels.astype(int)))
    # print(train_ds.output_shapes, test_ds.output_shapes)
    return train_ds, test_ds


if __name__ == '__main__':
    args = arg_parse()
    run_mnist_eager(args)'''
