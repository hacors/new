import tensorflow as tf

Dirsave = r'D:\myfile\data\DMLforPRID\model\logs'
# Dirsave = r'/root/data/DMLforPRID/model/logs'


class cnn():  # 只是生成模型，确定计算方式，而与数据无关
    def __init__(self, convshape, fullconnshape, inputdata, cnn_name):
        with tf.name_scope(cnn_name):
            self.all_data = tf.squeeze(inputdata, [1, 2])
            with tf.name_scope('initial'):
                self.conv_wbs = []
                self.full_conn_wbs = []
                for count, conv_stru in enumerate(convshape):
                    self.conv_wbs.append(self.initial_wbs(conv_stru, 'conv_%s' % count))
                for count, full_conn_stru in enumerate(fullconnshape):
                    self.full_conn_wbs.append(self.initial_wbs(full_conn_stru, 'fullcon_%s' % count))
            self.output = self.connect_all()

    def initial_wbs(self, myshape, name):
        with tf.name_scope(name):
            weights = tf.get_variable('%s_ws' % name, shape=myshape, dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=0.1, seed=1830801))
            biases = tf.get_variable('%s_bs' % name, shape=[
                                     myshape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        return [weights, biases]

    def conv_over(self, input_val):  # 实现卷积，SAME代表卷积时周围补充0，VALID表示不补充缩小
        last_activ = input_val  # 每一次的输入数据结构
        for count, layer in enumerate(self.conv_wbs):  # 需要事先初始化，layer[0]代表权重，layer[1]代表偏移
            with tf.name_scope('conv_%s' % count):
                logits = tf.nn.conv2d(last_activ, layer[0], strides=[1, 1, 1, 1],
                                      padding="SAME")  # 实现一次卷积计算，strides直接对应卷积核的维度的步长
                logits = tf.add(logits, layer[1])  # 偏移
                logits = tf.nn.leaky_relu(logits)  # 激活，改了
                # logits = tf.layers.batch_normalization(logits, axis=3, trainable=True)
                pool = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="VALID")
                last_activ = pool
        return last_activ

    def full_activation(self, input_val):  # 在卷积的基础上实现全连接
        last_activ = input_val
        for count, layer in enumerate(self.full_conn_wbs):
            with tf.name_scope('fullcon_%s' % count):
                logits = tf.matmul(last_activ, layer[0])
                logits = tf.add(logits, layer[1])
                # logits = tf.nn.relu(logits) #必须去除relu层
                last_activ = logits
        return last_activ

    def connect_all(self):
        conv_output = self.conv_over(self.all_data)  # 注意[0]位置的数据代表batchs大小
        conv_reshaped = tf.layers.flatten(conv_output)
        return self.full_activation(conv_reshaped)


class scnn():
    def __init__(self, convshape, fullconnshape, inputdata, scnn_name, labels, share):
        with tf.variable_scope(scnn_name):
            self.scnn_labels = labels
            self.all_data = inputdata
            if share:
                with tf.variable_scope('share', reuse=tf.AUTO_REUSE):
                    self.cnn_0 = cnn(convshape, fullconnshape, self.splitdata(self.all_data, 0), 'cnn_0')
                    self.cnn_1 = cnn(convshape, fullconnshape, self.splitdata(self.all_data, 1), 'cnn_1')
                self.cosin = self.get_cosin(self.cnn_0.output, self.cnn_1.output)
                self.loss = self.get_loss(self.scnn_labels, self.cosin)
            else:
                with tf.variable_scope('noshare_0'):
                    self.cnn_0 = cnn(convshape, fullconnshape, self.splitdata(self.all_data, 0), 'cnn_0')
                with tf.variable_scope('noshare_1'):
                    self.cnn_1 = cnn(convshape, fullconnshape, self.splitdata(self.all_data, 1), 'cnn_1')
                self.cosin = self.get_cosin(self.cnn_0.output, self.cnn_1.output)
                self.loss = self.get_loss(self.scnn_labels, self.cosin)

    def get_cosin(self, vector_a, vector_b):
        with tf.name_scope('cosin'):
            local_a = vector_a
            local_b = vector_b
            squar_sum_a = tf.sqrt(tf.reduce_sum(tf.square(local_a), 1))
            squar_sum_b = tf.sqrt(tf.reduce_sum(tf.square(local_b), 1))
            dot_a_b = tf.reduce_sum(tf.multiply(local_a, local_b), axis=1)
            cosin = tf.divide(dot_a_b, tf.multiply(squar_sum_a, squar_sum_b))
        return cosin

    def get_loss(self, tlabels, tcosins):
        with tf.name_scope('ave_loss'):
            # loss = tf.reduce_mean(tf.log(tf.exp(-2*tf.multiply(tlabels, tcosins)+1)), axis=0)
            loss = tf.reduce_mean(tf.log(tf.exp(-2*tf.multiply(tlabels, tcosins))+1), axis=0)
        return loss

    def splitdata(self, inputdata, part):
        with tf.name_scope('cnn_data%s' % part):
            temptensor = tf.slice(inputdata, [0, 0, part, 0, 0, 0], [-1, -1, 1, -1, -1, -1])
            return temptensor


class triple_scnn():
    def __init__(self, convshape, fullconnshape, inputshape, split_scnn_name, share):
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(tf.float32, shape=[None], name='labels')
        with tf.name_scope('all_data'):
            self.all_data = tf.placeholder(
                tf.float32, shape=[None, 3, 2, inputshape[1], inputshape[2], inputshape[3]], name='datas')
        self.scnn_0 = scnn(convshape, fullconnshape, self.splitdata(self.all_data, 0), 'scnn_0', self.labels, share)
        self.scnn_1 = scnn(convshape, fullconnshape, self.splitdata(self.all_data, 1), 'scnn_1', self.labels, share)
        self.scnn_2 = scnn(convshape, fullconnshape, self.splitdata(self.all_data, 2), 'scnn_2', self.labels, share)
        self.cosinsums = self.cosin_sum(self.scnn_0.cosin, self.scnn_1.cosin, self.scnn_2.cosin)
        self.losssums = self.loss_sum(self.scnn_0.loss, self.scnn_1.loss, self.scnn_2.loss)

    def cosin_sum(self, cos_h, cos_b, cos_l):
        with tf.name_scope('cosin_sum'):
            sums = tf.add_n([cos_h, cos_b, cos_l])
        return sums

    def loss_sum(self, loss_h, loss_b, loss_l):
        with tf.name_scope('loss_sum'):
            sums = tf.add_n([loss_h, loss_b, loss_l])
        return sums

    def splitdata(self, inputdata, part):
        with tf.name_scope('scnn_data%s' % part):
            temptensor = tf.slice(inputdata, [0, part, 0, 0, 0, 0], [-1, 1, -1, -1, -1, -1])
            return temptensor


# 测试部分
if __name__ == '__main__':
    trip_scnn = triple_scnn([[7, 7, 3, 32], [5, 5, 32, 48]], [[6912, 500]], [None, 48, 48, 3], 'VIPeR', True)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(Dirsave, graph=sess.graph)
# tensorboard --logdir D:\myfile\data\DMLforPRID\model\logs --port 6007
