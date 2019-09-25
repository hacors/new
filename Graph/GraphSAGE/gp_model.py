import tensorflow as tf
KERAS = tf.keras


# 输入为图的邻居结点结构信息，输出为选取的某些结点的邻居（从各个结点邻居中随机）
class UniformNeighborSampler(KERAS.Model):
    def __init__(self, node_adjlist):
        super().__init__(name='UniformNeighborSampler')
        self.node_adjlist=node_adjlist
        self.embed_layer=KERAS.Input(shape=self.node_adjlist.shape())


class GCNAggregator():
    pass


class NeighborSampler():
    pass


class SampleAndAggregate(object):
    def __init__(self):
        self.aggregator = GCNAggregator()

    def sample(self):
        pass
