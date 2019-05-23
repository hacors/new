import os
import time

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
tf.enable_eager_execution()

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)

SHUFFLE_SIZE = 1000
BATCH_SIZE = 256
INPUT_DIM = 100
EPOCH = 80
IMAGES_NUM = 16
print('version: tensorflow %s,keras %s\n' % (tf.VERSION, tf.keras.__version__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_datas():
    (train_images_orig, train_labels_orig), (test_images_orig, test_labels_orig) = tf.keras.datasets.mnist.load_data()
    train_images_reshaped = train_images_orig[..., tf.newaxis]
    train_images_casted = tf.cast((train_images_reshaped-127.5)/127.5, tf.float32)
    train_images_batched = tf.data.Dataset.from_tensor_slices(train_images_casted).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    return train_images_batched


'''
def discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model


def generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model
'''


def generator(input_shape=(INPUT_DIM, 1), conv_list=[16, 16, 1], dens_list=[128, 784]):
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


def discriminator(input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[128, 1]):
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
    real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_results), real_results)
    fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_results), fake_results)
    return real_loss+fake_loss


def generator_loss(fake_results):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_results), fake_results)


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


def save_tempdatas(model, epoch, test_input):
    predictions = model(test_input, training=False)
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('Temp/graphs/image_{}.png'.format(epoch))
    plt.close()


def train():
    gener = generator()
    discri = discriminator()
    g_opti = tf.train.AdamOptimizer(learning_rate=1e-3)
    d_opti = tf.train.AdamOptimizer(learning_rate=1e-3)

    checkpoint_dir = 'Temp/models'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=g_opti, discriminator_optimizer=d_opti, generator=gener, zdiscriminator=discri)
    vectors_show_images = tf.random_normal([IMAGES_NUM, INPUT_DIM])

    datas = get_datas()
    for epoch in range(EPOCH):
        start = time.time()
        for data in datas:
            train_step(data, BATCH_SIZE, INPUT_DIM, gener, discri, g_opti, d_opti)
        save_tempdatas(gener, epoch, vectors_show_images)
        print('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
        if (epoch+1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


train()
