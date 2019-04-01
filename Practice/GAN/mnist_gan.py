import os

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('version: tensorflow %s,keras %s\n' % (tf.VERSION, tf.keras.__version__))

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)


def get_train_iter():
    with tf.name_scope('train_iter'):
        (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
        train_image = tf.cast(train_image[..., tf.newaxis]/255, tf.float32)
        train_label = tf.one_hot(train_label, 10, 1.0, 0.0)
        train_set = tf.data.Dataset.from_tensor_slices((train_image, train_label))
        train_set = train_set.shuffle(1000).batch(32)
        train_iter = train_set.make_one_shot_iterator()
        return train_iter


def def_discrimanator(input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[10, 10]):
    with tf.variable_scope('discriminator'):
        input_data = keras.layers.Input(shape=input_shape)
        digits = input_data
        for dim in conv_list:
            digits = keras.layers.Conv2D(dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
            digits = keras.layers.MaxPool2D()(digits)
        digits = keras.layers.Flatten()(digits)
        for dim in dens_list:
            digits = keras.layers.Dense(dim, activation=keras.layers.LeakyReLU(), kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        prediction = keras.layers.Softmax()(digits)
        model = keras.Model(inputs=input_data, outputs=prediction)
        return(model)


def do_train():
    model = def_discrimanator()
    opti = tf.train.AdamOptimizer()
    iters = get_train_iter()
    while True:
        image, label = iters.get_next()
        with tf.GradientTape() as tape:
            logits = model(image)
            loss = tf.losses.softmax_cross_entropy(label, logits)
            print(loss.numpy())
        grads = tape.gradient(loss, model.variables)
        opti.apply_gradients(zip(grads, model.variables))


do_train()
