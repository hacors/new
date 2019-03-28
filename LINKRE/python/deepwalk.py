import os
import networkx as nx
director = 'LINKRE/python/temp/deepwalk.txt'


def getresult(net: nx.Graph):
    alledge = net.edges
    print(alledge)
    return


if __name__ == '__main__':
    mygraph = nx.Graph()
    mygraph.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 5), (2, 3), (2, 4), (2, 5), (2, 5)])
    mydeepwalk = getresult(mygraph)
