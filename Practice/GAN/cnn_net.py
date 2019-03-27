import tensorflow as tf

tf.enable_eager_execution()


class Model_dis(tf.keras.Model):
    def __init__(self):
        super(tf.keras.Model, self).__init__()
        layers = tf.keras.layers
        random_init = tf.random_normal_initializer(stddev=0.1)
        cons_int = tf.constant_initializer(value=0.1)
        self.conv_1 = layers.Conv3D(16, [1, 5, 5], padding='same', kernel_regularizer=random_init, bias_initializer=cons_int, activation=layers.LeakyReLU())
        self.pool_1 = layers.MaxPool2D()
        self.conv_2 = layers.Conv3D(16, [16, 5, 5], padding='same', kernel_regularizer=random_init, bias_initializer=cons_int, activation=layers.LeakyReLU())
        self.pool_2 = layers.MaxPool2D()
        self.flat = layers.Flatten()
        self.dense_1 = layers.Dense(10, kernel_initializer=random_init, bias_initializer=cons_int)
        self.dense_2 = layers.Dense(10, kernel_initializer=random_init, bias_initializer=cons_int)

    def call(self, input_data):
        result = self.conv_1(input_data)


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

    def net_design(self, input_data):
        digits = tf.keras.layers.Dense
