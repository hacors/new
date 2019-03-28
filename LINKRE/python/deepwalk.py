import os
import networkx as nx


def getresult(net: nx.Graph):
    alledge=net.edges

    return


if __name__ == '__main__':
    mygraph = nx.Graph()
    mygraph.add_edges_from([('1', '3'), ('1', '4'), ('1', '5'), ('1', '5'), ('2', '3'), ('2', '4'), ('2', '5'), ('2', '5')])
    mydeepwalk = deepwalk(mygraph)
    model = mydeepwalk.getresult(10, 0.2)
    print(model.similarity('1', '2'))
    print(model.similarity('1', '3'))
    print(model.similarity('3', '4'))
