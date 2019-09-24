import numpy as np
import networkx as nx
np.random.seed(666)


class MinibatchIterator(object):
    def __init__(self, graph: nx.Graph,  batch_size=100, max_degree=25):
        self.graph = graph
        # 定义图的基本信息
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.node_num = self.graph.number_of_nodes()
        self.edge_num = self.graph.number_of_edges()
        self.batch_num = self.edge_num // self.batch_size + 1  # 总的batch数
        # 构造结构化的数据，将图的结构化数据转换为矩阵形式的邻居数据
        self.node_adjlist, self.node_deg = self.construct_adj()
        # 构造训练数据，需要将edges和nodes打乱
        self.shuffle()

    def shuffle(self):
        self.rd_nodelist = np.random.permutation(self.graph.nodes())
        self.rd_edgelist = np.random.permutation(self.graph.edges())
        self.batch_index = 0  # 记录当前产生的batch的编号

    def construct_adj(self):  # 为每一个结点选取固定数目的邻居，并且同时返回这个结点的真实邻居数
        adj = self.node_num * np.ones((self.node_num, self.max_degree), dtype=np.int32)  # 如果没有邻居，那么采样到的邻居就是不存在的标号
        deg = np.zeros(self.node_num, dtype=np.int32)
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


class EdgeMinibatchIterator(MinibatchIterator):
    def __init__(self, graph: nx.Graph,  batch_size=100, max_degree=25):
        super().__init__(graph, batch_size, max_degree)

    def next_minibatch_feed_dict(self):  # 每次调用获取一定量的数据，但是总的调用次数是确定的
        start_idx = self.batch_index * self.batch_size
        self.batch_index += 1
        end_idx = min(start_idx + self.batch_size, len(self.rd_edgelist))
        batch_edges = self.rd_edgelist[start_idx: end_idx]
        # 需要将batch_edges转换为dict输出
        batch_v0, batch_v1 = [], []
        for v0, v1 in batch_edges:
            batch_v0.append(v0)
            batch_v1.append(v1)
        feed_dict = {}
        feed_dict['batch_size'] = len(batch_edges)
        feed_dict['batch_v0'] = batch_v0
        feed_dict['batch_v1'] = batch_v1
        return feed_dict


class NodeMinibatchIterator(MinibatchIterator):
    def __init__(self, graph: nx.Graph, label_map, label_class, batch_size=100, max_degree=25):
        super().__init__(graph, batch_size, max_degree)
        self.label_map = label_map
        self.label_class = label_class

    def get_label_vec(self, node):  # 获取一个结点的label（必须为一个向量）
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.label_class))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def next_minibatch_feed_dict(self):  # 每次调用获取一定量的数据，但是总的调用次数是确定的
        # 需要返回一个node的序列，以及node对应label（以编码形式存在）的序列
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        batch_labels = np.vstack([self.get_label_vec(node) for node in batch_nodes], axis=1)
        feed_dict = {}
        feed_dict['batch_size'] = len(batch_nodes)
        feed_dict['batch_nodes'] = batch_nodes
        feed_dict['batch_labels'] = batch_labels
        return feed_dict
