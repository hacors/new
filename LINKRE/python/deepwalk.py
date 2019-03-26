import random as rd

from gensim.models import Word2Vec
import networkx as nx


class deepwalk():
    def __init__(self, network: nx.Graph):
        self.network = network

    def one_walk(self, node, network: nx.Graph, length, alpha):
        path = list([node])
        while len(path) < length:
            cur = path[-1]
            if rd.random() >= alpha:
                neighbors = [nei for nei in network.neighbors(cur)]
                choice = rd.choice(neighbors)
                path.append(choice)
            else:
                path.append(path[0])
        path = [str(node) for node in path]
        return path

    def all_walk(self, repe, network: nx.Graph, length, alpha):
        walks = list()
        nodes = [node for node in network.nodes()]
        for trave in range(repe):
            rd.shuffle(nodes)
            for node in nodes:
                path = self.one_walk(node, network, length, alpha)
                walks.append(path)
        return walks

    def train(self, walks):
        model = Word2Vec(sentences=walks)
        return model

    def getresult(self, repe, alpha):
        length = len(self.network.nodes())*2
        walks = self.all_walk(repe, self.network, length, alpha)
        model = self.train(walks)
        return model


if __name__ == '__main__':
    mygraph = nx.Graph()
    mygraph.add_edges_from([('1', '3'), ('1', '4'), ('1', '5'), ('1', '5'), ('2', '3'), ('2', '4'), ('2', '5'), ('2', '5')])
    mydeepwalk = deepwalk(mygraph)
    model = mydeepwalk.getresult(10, 0.2)
    print(model.similarity('1', '2'))
    print(model.similarity('1', '3'))
    print(model.similarity('3', '4'))
