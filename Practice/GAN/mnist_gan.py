import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
print('version: tensorflow %s,keras %s' % (tf.VERSION, tf.keras.__version__))

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)


def data():
    with tf.name_scope('data'):
        (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
        train_data = tf.cast(tf.expand_dims(train_data, -1)/255, tf.float32)
        train_label = tf.cast(train_label, tf.int32)
        train_set = tf.data.Dataset.from_tensor_slices((train_data, train_label))
        train_set = train_set.shuffle(1000).batch(32)
        test_data = tf.cast(tf.expand_dims(test_data, -1)/255, tf.float32)
        test_label = tf.cast(test_label, tf.int32)
        test_set = tf.data.Dataset.from_tensor_slices((test_data, test_label))
        test_set = test_set.shuffle(1000).batch(32)
        return train_set, test_set


def discrimanator(input_shape, conv_list: list, dens_list: list):
    with tf.variable_scope('discriminator'):
        input_data = keras.layers.Input(shape=input_shape)
        digits = input_data
        for dim in conv_list:
            digits = keras.layers.Conv3D(dim, (5, 5, 1), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
            digits = keras.layers.MaxPool2D()(digits)
        digits = keras.layers.Flatten()(digits)
        for dim in dens_list:
            digits = keras.layers.Dense(dim, activation=keras.layers.LeakyReLU, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
        prediction = keras.layers.Softmax()
        model = keras.Model(inputs=input_data, outputs=prediction)
        return model


dataset = data()
for data1 in dataset:
    print(data1[0])
'''
mydis = discrimanator((28, 28, 1), [12, 12], [10, 10])
'''
