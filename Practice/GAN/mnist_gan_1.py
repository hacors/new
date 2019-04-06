import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)
print('version: tensorflow %s,keras %s\n' % (tf.VERSION, tf.keras.__version__))

(train_images_orig, train_labels_orig), (test_images_orig,
                                         test_labels_orig) = tf.keras.datasets.mnist.load_data()
train_images_casted = tf.cast(train_images_orig[..., tf.newaxis]/255, tf.float32)
print(train_images_casted.shape)


def discrimanator(input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[128, 16, 1]):
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
    prediction = digits
    model = keras.Model(inputs=input_data, outputs=prediction)
    return(model)


discri = discrimanator()
print(discri)


def generator(input_shape=(10, 1), conv_list=[16, 16, 1], dens_list=[784, 3136]):
    input_data = keras.layers.Input(shape=input_shape)
    digits = input_data
    for dim in dens_list:
        digits = keras.layers.Dense(
            dim, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.LeakyReLU()(digits)
    digits = keras.layers.Reshape((56, 56, 1))
    for dim in conv_list:
        digits = keras.layers.Conv2D(
            dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        digits = keras.layers.MaxPool2D()(digits)
    prediction = digits
    model = keras.Model(inputs=input_data, outputs=prediction)
    return(model)


gener = generator()
print(gener)
