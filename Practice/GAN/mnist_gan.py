import os
import shutil

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


class simplegan():
    def __init__(self):
        self.discri = self.def_discrimanator()
        self.do_train()
        self.store()

    def get_train_iter(self):
        with tf.name_scope('train_iter'):
            (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
            train_image = tf.cast(train_image[..., tf.newaxis]/255, tf.float32)
            train_label = tf.one_hot(train_label, 10, 1.0, 0.0)
            train_set = tf.data.Dataset.from_tensor_slices((train_image, train_label))
            train_set = train_set.shuffle(1000).batch(256)
            train_iter = train_set.make_one_shot_iterator()
            return(train_iter)

    def def_discrimanator(self, input_shape=(28, 28, 1), conv_list=[16, 16], dens_list=[10, 10]):
        with tf.variable_scope('discriminator'):
            input_data = keras.layers.Input(shape=input_shape)
            digits = input_data
            for dim in conv_list:
                digits = keras.layers.Conv2D(dim, (5, 5), padding='same', kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
                digits = keras.layers.MaxPool2D()(digits)
            digits = keras.layers.Flatten()(digits)
            for dim in dens_list:
                digits = keras.layers.Dense(dim, activation=keras.layers.LeakyReLU(), kernel_initializer=ker_init, bias_initializer=bia_init)(digits)
            prediction = digits
            model = keras.Model(inputs=input_data, outputs=prediction)
            return(model)

    def do_train(self):
        model = self.discri
        iters = self.get_train_iter()
        opti = tf.train.AdamOptimizer(learning_rate=0.001)
        try:
            while True:
                image, label = iters.get_next()
                with tf.GradientTape() as tape:
                    logits = model(image)
                    loss = tf.losses.softmax_cross_entropy(label, logits)
                    print(loss.numpy())
                    grads = tape.gradient(loss, model.variables)
                opti.apply_gradients(zip(grads, model.variables))
        except tf.errors.OutOfRangeError:
            print('iter end')
        finally:
            print('train stop')

    def store(self):
        shutil.rmtree(model_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        builder.add_meta_graph_and_variables(self.discri, ['saved_graph'])
        builder.save()


mysimple = simplegan()
