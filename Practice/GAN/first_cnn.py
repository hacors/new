import os

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('version: tensorflow %s,keras %s\n' % (tf.VERSION, tf.keras.__version__))

ker_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
bia_init = tf.initializers.constant(value=0.1)

# model_dir = 'D:/myfile/data/eager/temp'
model_dir = 'Practice/GAN/model/temp'
board_dir = 'Practice/GAN/board'


class simple():
    def __init__(self):
        self.discri = self.def_discrimanator()
        self.train_iter, self.test_iter = self.get_data_iter()

    def get_data_iter(self):
        with tf.name_scope('train_iter'):
            (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
            train_image = train_image[:10000]
            train_label = train_label[:10000]
            train_image = tf.cast(train_image[..., tf.newaxis]/255, tf.float32)
            train_label = tf.one_hot(train_label, 10, 1.0, 0.0)
            train_set = tf.data.Dataset.from_tensor_slices((train_image, train_label))
            train_set = train_set.shuffle(1000).batch(500)
            train_iter = train_set.make_one_shot_iterator()

            test_image = tf.cast(test_image[..., tf.newaxis]/255, tf.float32)
            test_label = tf.one_hot(test_label, 10, 1.0, 0.0)
            test_set = tf.data.Dataset.from_tensor_slices((test_image, test_label))
            test_set = test_set.shuffle(1000).batch(512)
            test_iter = test_set.make_one_shot_iterator()
            return train_iter, test_iter

    def def_discrimanator(self, input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[10, 10]):
        with tf.variable_scope('discriminator'):
            input_data = keras.layers.Input(shape=input_shape)
            digits = input_data
            for dim in conv_list:
                digits = keras.layers.Conv2D(dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
                digits = keras.layers.MaxPool2D()(digits)
            digits = keras.layers.Flatten()(digits)
            for dim in dens_list:
                digits = keras.layers.Dense(dim, kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
                digits = keras.layers.LeakyReLU()(digits)
            prediction = digits
            model = keras.Model(inputs=input_data, outputs=prediction)
            return model

    def do_train(self):
        model = self.discri
        iters = self.train_iter
        opti = tf.train.AdamOptimizer(learning_rate=0.001)
        board_write = tf.contrib.summary.create_file_writer(board_dir)
        board_write.as_default()
        tf.contrib.summary.always_record_summaries()
        try:
            while True:
                images, labels = iters.get_next()
                with tf.GradientTape() as tape:
                    logits = model(images)
                    loss = tf.losses.softmax_cross_entropy(labels, logits)
                    tf.contrib.summary.scalar('loss', loss)
                    print(loss.numpy())
                    grads = tape.gradient(loss, model.variables)
                opti.apply_gradients(zip(grads, model.variables))
        except tf.errors.OutOfRangeError:
            print('iters end')
        finally:
            print('train stop')
            model.save_weights(model_dir)

    def load_and_test(self):
        test_model = self.def_discrimanator()
        test_model.load_weights(model_dir)
        iters = self.test_iter
        num_accus = 0
        num_test = 0
        try:
            while True:
                images, labels = iters.get_next()
                logits = test_model(images)
                temp_accus, temp_test = self.accurate(labels, logits)
                num_accus += temp_accus
                num_test += temp_test
        except tf.errors.OutOfRangeError:
            print('iters end')
        finally:
            print('test end')
            print(num_accus/num_test)

    def accurate(self, labels, logits):
        labels = tf.arg_max(labels, 1, output_type=tf.int64)
        logits = tf.arg_max(logits, 1, output_type=tf.int64)
        equals = tf.cast(tf.equal(logits, labels), tf.int32)
        results = int(tf.reduce_sum(equals))
        return results, int(labels.shape[0])


mysimple = simple()
mysimple.do_train()
mysimple.load_and_test()
# tensorboard --logdir D:\code\Practice\GAN\board --port 6006
