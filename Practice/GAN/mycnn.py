import tensorflow as tf


class mycnn():  # 只是生成模型，确定计算方式，而与数据无关
    '''
    convshape: [[7, 7, 16], [5, 5, 16]]
    fullconnshape: [[16], [10]]
    '''

    def __init__(self, convshape, fullconnshape, inputdata: tf.Tensor, netname='default'):
        with tf.name_scope(netname):
            convshape = self.stru_change(convshape, inputdata.shape[-1].value)
            self.conv_wbs = list()
            for count, conv_stru in enumerate(convshape):
                self.conv_wbs.append(self.initial_wbs(conv_stru, 'conv_%s' % count))
            conv_output = self.conv_over(inputdata)
            conv_reshaped = tf.layers.flatten(conv_output)

            fullconnshape = self.stru_change(fullconnshape, conv_reshaped.shape[-1].value)
            self.full_conn_wbs = list()
            for count, full_conn_stru in enumerate(fullconnshape):
                self.full_conn_wbs.append(self.initial_wbs(full_conn_stru, 'fullcon_%s' % count))
            self.output = self.full_activation(conv_reshaped)

    def stru_change(self, stru_list, first_data):
        data = first_data
        for struct in stru_list:
            struct.insert(-1, data)
            data = struct[-1]
        return stru_list

    def initial_wbs(self, myshape, name):
        with tf.name_scope(name):
            weights = tf.get_variable('%s_ws' % name, shape=myshape, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1, seed=1830801))
            biases = tf.get_variable('%s_bs' % name, shape=[myshape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        return [weights, biases]

    def conv_over(self, input_val):  # 实现卷积，SAME代表卷积时周围补充0，VALID表示不补充缩小
        last_activ = input_val  # 每一次的输入数据结构
        for count, layer in enumerate(self.conv_wbs):  # 需要事先初始化，layer[0]代表权重，layer[1]代表偏移
            with tf.name_scope('conv_%s' % count):
                logits = tf.nn.conv2d(last_activ, layer[0], strides=[1, 1, 1, 1], padding="SAME")  # 实现一次卷积计算，strides直接对应卷积核的维度的步长
                logits = tf.add(logits, layer[1])  # 偏移
                logits = tf.nn.leaky_relu(logits)  # 激活，改了
                pool = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
                last_activ = pool
        return last_activ

    def full_activation(self, input_val):  # 在卷积的基础上实现全连接
        last_activ = input_val
        for count, layer in enumerate(self.full_conn_wbs):
            with tf.name_scope('fullcon_%s' % count):
                logits = tf.matmul(last_activ, layer[0])
                logits = tf.add(logits, layer[1])
                logits = tf.nn.sigmoid(logits)
                last_activ = logits
        return last_activ


if __name__ == '__main__':
    pass
