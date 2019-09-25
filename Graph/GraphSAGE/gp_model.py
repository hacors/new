import tensorflow as tf
KERAS = tf.keras


# 输入为图的邻居结点结构信息，输出为选取的某些结点的邻居（从各个结点邻居中随机）
class UniformNeighborSampler(KERAS.Model):
    def __init__(self, node_adjlist):
        super().__init__(name='UniformNeighborSampler')
        self.node_adjlist = node_adjlist
        self.embed_layer =
        self.


class MyLayer(KERAS.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GCNAggregator():
    pass


class NeighborSampler():
    pass


class SampleAndAggregate(object):
    def __init__(self):
        self.aggregator = GCNAggregator()

    def sample(self):
        pass
