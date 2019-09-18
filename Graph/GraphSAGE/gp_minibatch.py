import numpy as np
import networkx as nx
np.random.seed(666)


class EdgeMinibatchIterator(object):
    '''
    max_degree为邻居采样数
    '''

    def __init__(self, graph: nx.Graph,  batch_size=100, max_degree=25):
        self.graph = graph
        self.batch_size = batch_size
        self.max_degree = max_degree
        # 构造结构化的数据
        self.adj, self.deg = self.construct_adj()
        # 构造训练数据，需要将edges和nodes打乱
        self.nodes = np.random.permutation(self.graph.nodes())
        self.edges = np.random.permutation(self.graph.edges())

        self.batch_num = 0  # 记录当前产生的batch的编号

        pass

    def construct_adj(self):  # 为每一个结点选取固定数目的邻居，并且同时返回这个结点的真实邻居数
        adj = self.graph.number_of_nodes() * np.ones((self.graph.number_of_nodes(), self.max_degree), dtype=np.int32)  # 如果没有邻居，那么采样到的邻居就是不存在的标号
        deg = np.zeros(self.graph.number_of_nodes(), dtype=np.int32)
        for nodeid in self.graph.nodes():
            deg[nodeid] = self.graph.degree(nodeid)
            neighbors = np.array([nei_id for nei_id in self.graph.neighbors(nodeid)])
            if self.graph.degree(nodeid) == 0:
                continue
            elif self.graph.degree(nodeid) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            else:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj, deg

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.edges))
        batch_edges = self.edges[start_idx: end_idx]
        return self.batch_feed_dict(batch_edges)
