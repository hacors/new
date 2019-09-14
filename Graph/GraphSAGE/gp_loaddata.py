import random
import json
import networkx as nx
from sklearn.preprocessing import StandardScaler
WALK_LEN = 5
N_WALKS = 50


def load_data(graph_dir):
    graph_data = json.load(open(graph_dir))  # 加载使用json文件存储的图数据
    graph = nx.readwrite.json_graph.node_link_graph(graph_data)  # 将图数据转换为图结构，其中每一个结点都有一个长度为50的feat，一个长度为121的class
    # 在图中抽离出features矩阵和labels矩阵，并对features做归一化
    features = []
    labels = []
    for node_id in graph.nodes:
        features.append(graph.node[node_id]['feature'])
        labels.append(graph.node[node_id]['label'])
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    return graph, features, labels


def run_random_walks(walks_dir, graph):
    # 对所有的结点做随机游走，并且存储dict（每一个结点游走结果）
    walks = {}
    for node in graph.nodes:
        walks[node] = []
        if graph.degree(node) == 0:
            continue
        for i in range(N_WALKS):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(list(graph.neighbors(curr_node)))
                if curr_node != node:
                    walks[node].append(curr_node)
                curr_node = next_node
    with open(walks_dir, mode='w') as outfile:
        json.dump(walks, outfile)


if __name__ == "__main__":
    director = r'Datasets\Temp\Graphsage_data'
    graph, features, labels = load_data(director+r'\toy-ppi-G.json')
    run_random_walks(director+r'\toy-ppi-walks.json', graph)
