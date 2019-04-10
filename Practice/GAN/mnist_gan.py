import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

tf.enable_eager_execution()

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)

BATCH_SIZE = 256
NOISE_DIM = 100
EPOCH = 50
print('version: tensorflow %s,keras %s\n' % (tf.VERSION, tf.keras.__version__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
(train_images_orig, train_labels_orig), (test_images_orig, test_labels_orig) = tf.keras.datasets.mnist.load_data()
train_images_casted = tf.cast(train_images_orig[..., tf.newaxis]/255, tf.float32)
datas_scale = int(train_images_casted.shape[0].value)
random_datas_casted = tf.random_normal([datas_scale, 10], dtype=tf.float32)
train_images_iter = tf.data.Dataset.from_tensor_slices(train_images_casted).shuffle(1000).batch(10).make_one_shot_iterator()
random_datas_iter = tf.data.Dataset.from_tensor_slices(random_datas_casted).shuffle(1000).batch(10).make_one_shot_iterator()
'''


def generator_model(input_shape=(10, 1), conv_list=[16, 16, 1], dens_list=[128, 784]):
    input_data = keras.layers.Input(shape=input_shape)
    digits = input_data
    digits = keras.layers.Flatten()(digits)
    for dim in dens_list:
        digits = keras.layers.Dense(
            dim, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.BatchNormalization()(digits)
        digits = keras.layers.LeakyReLU()(digits)
    digits = keras.layers.Reshape((28, 28, 1))(digits)
    for dim in conv_list:
        digits = keras.layers.Conv2D(
            dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.BatchNormalization()(digits)
        digits = keras.layers.LeakyReLU()(digits)
    prediction = keras.layers.Activation(activation='sigmoid')(digits)
    model = keras.Model(inputs=input_data, outputs=prediction)
    return model


def discriminator_model(input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[128, 1]):
    input_data = keras.layers.Input(shape=input_shape)
    digits = input_data
    for dim in conv_list:
        digits = keras.layers.Conv2D(
            dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.BatchNormalization()(digits)
        digits = keras.layers.LeakyReLU()(digits)
        digits = keras.layers.MaxPool2D()(digits)
    digits = keras.layers.Flatten()(digits)
    for dim in dens_list:
        digits = keras.layers.Dense(
            dim, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.BatchNormalization()(digits)
        digits = keras.layers.LeakyReLU()(digits)
    prediction = keras.layers.Activation(activation='sigmoid')(digits)
    model = keras.Model(inputs=input_data, outputs=prediction)
    return model


def discriminator_loss(real_results, fake_results):
    real_loss = tf.losses.log_loss(tf.ones_like(real_results), real_results)
    fake_loss = tf.losses.log_loss(tf.zeros_like(fake_results), fake_results)
    return real_loss+fake_loss


def generator_loss(fake_results):
    return tf.losses.log_loss(tf.ones_like(fake_results), fake_results)


def train_step(real_images, batch_size, z_dim, generator, discriminator, generator_opti, discriminator_opti):
    noise = tf.random_normal([batch_size, z_dim])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        real_results = discriminator(real_images, training=True)
        fake_results = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_results)
        d_loss = discriminator_loss(real_results, fake_results)
    d_gradiens = g_tape.gradient(g_loss, generator.variables)
    g_gradiens = d_tape.gradient(d_loss, discriminator.variables)
    generator_opti.apply_gradients(zip(d_gradiens, generator.variables))
    discriminator_opti.apply_gradients(zip(g_gradiens, discriminator.variables))


def showimages(images):
    datas = images.numpy()*255
    plt.ion()
    data = datas[0]
    data = data.reshape((28, 28)).astype(np.int)
    plt.imshow(data, cmap='Greys')
    plt.pause(0.1)


'''
discri = discriminator()
gener = generator()
real_images_iter = train_images_iter
fake_vectors_iter = random_datas_iter
d_opti = tf.train.AdamOptimizer(learning_rate=0.0003)
g_opti = tf.train.AdamOptimizer(learning_rate=0.0003)
step = 0
try:
    while True:
        step += 1
        vectors = fake_vectors_iter.get_next()
        real_images = real_images_iter.get_next()
        with tf.GradientTape() as d_tape:
            d_fake_images = gener(vectors)
            d_fake_results = discri(d_fake_images)
            d_real_results = discri(real_images)
            d_all_results = tf.concat([d_fake_results, d_real_results], axis=0)
            d_all_labels = tf.concat([tf.zeros_like(d_fake_results), tf.ones_like(d_real_results)], axis=0)
            d_loss = tf.reduce_mean(tf.losses.log_loss(d_all_labels, d_all_results))
            print('d_loss:', d_loss.numpy(), end='  ')
            d_grads = d_tape.gradient(d_loss, discri.variables)
            d_opti.apply_gradients(zip(d_grads, discri.variables))
        if(step % 10 == 0):
            showimages(d_fake_images)

        with tf.GradientTape() as g_tape:
            g_fake_images = gener(vectors)
            g_fake_results = discri(g_fake_images)
            g_loss = tf.reduce_mean(tf.losses.log_loss(tf.ones_like(g_fake_results), g_fake_results))
            print('g_loss:', g_loss.numpy())
            g_grads = g_tape.gradient(g_loss, gener.variables)
            g_opti.apply_gradients(zip(g_grads, gener.variables))
except tf.errors.OutOfRangeError:
    print('iters end')
finally:
    print('train stop')
'''
