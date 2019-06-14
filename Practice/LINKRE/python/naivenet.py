import math
import random as rd

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import support as sup
from deepwalk import walks
from deepwalk import skipgram
edge_director = sup.ROOT + '/temp/edges'
vector_director = sup.ROOT + '/temp/vectors'


class mynetwork():  # 图类
    '''
    gratps:图的类型
    nodenum:结点数量
    '''

    def __init__(self, gratps, nodenum=-1):  # 初始化，-1表示默认提取所有点
        self.types = sup.net_types(gratps)
        self.network = self.get_networkx(self.types, nodenum)
        self.setallparas()

    def get_networkx(self, nettypes, nodenum):
        '''
        nodenum只在BA有作用
        '''
        if nettypes.name == 'ba':
            linknum = sup.net_paras['balinks']
            return(self.get_ba(nodenum, linknum))
        else:
            director = sup.ROOT + '/networks/%s' % nettypes.name
            return(self.get_fromfile(nodenum, director))

    def get_ba(self, nodenum, linknum):  # BA的实现，最小结点数为2
        assert(nodenum >= 2)
        network = nx.Graph()
        degreesum = 2
        degreelist = [1, 1]
        network.add_edge(0, 1)
        for adding in range(2, nodenum):  # 插入所有点
            nodeinputed = network.number_of_nodes()
            needlink = min(nodeinputed, linknum)
            degreeprob = (np.array(degreelist)/degreesum).tolist()
            choosed = np.random.choice(nodeinputed, size=needlink, replace=False, p=degreeprob)
            network.add_node(adding)
            for nodelinked in choosed:
                network.add_edge(adding, nodelinked)
                degreelist[nodelinked] += 1
            degreelist.append(needlink)
            degreesum += needlink*2
        return(network)

    def get_fromfile(self, nodenum, director):  # 读取网络
        network = nx.Graph()
        file = open(director, 'r')
        for lines in file.readlines():
            if(lines[0] == '#'):
                continue
            strlist = lines.rstrip().split(' ')
            intlist = list(map(int, strlist))
            network.add_edge(intlist[0], intlist[1])
        if(nodenum >= len(network.nodes) or nodenum == -1):
            return network
        else:  # 基于某一点开始遍历
            return self.netBFS(network, nodenum)

    def netBFS(self, network: nx.Graph, num):
        newnet = nx.Graph()
        allnodes = list(network.nodes)
        source = np.random.choice(allnodes, 1)
        count = 1
        frontiers = list(source)
        traves = list(source)
        while(frontiers):
            nexts = []
            for front in frontiers:
                for current in network.neighbors(front):
                    if current not in traves:
                        traves.append(current)
                        nexts.append(current)
                        tempneis = network.neighbors(current)
                        for node in tempneis:
                            if node in traves and network.has_edge(current, node):
                                newnet.add_edge(current, node)
                        count += 1
                        if(count >= num):
                            return newnet
            frontiers = nexts
        return newnet

    def shownet(self, network, name='default'):  # 简单展示网络
        plt.cla()
        position = nx.kamada_kawai_layout(network)
        edgelists = [(v0, v1)for (v0, v1)in network.edges if True]
        nx.draw_networkx_nodes(network, position, node_color='green', node_size=300)
        nx.draw_networkx_labels(network, position, font_size=8, font_color='white')
        nx.draw_networkx_edges(network, position, edgelist=edgelists, edge_color='black', width=1.0)
        direct = sup.ROOT + '/graphs/'+name+'.png'
        plt.savefig(direct)
        plt.close()

    def showimfo(self):
        network = self.network
        print(self.types)
        print('# nodenum:', len(network.nodes))
        print('# edgenum:', len(network.edges))

    def setallparas(self):  # 设定所有相关参数
        samax = sup.rec_paras['samax']
        lpmax = sup.rec_paras['lpmax']
        lpmult = sup.rec_paras['lpmult']
        dilipara1, dilipara2 = sup.mov_paras['dilipara1'], sup.mov_paras['dilipara2']  # 结点邻居度的和加载一起，但是度需要一定的修饰
        lparrays = self.getarrayspa(lpmax)
        dwmodels = self.get_deepwalk()
        for nodeindex in self.network.nodes:
            tempneis = self.network.neighbors(nodeindex)
            count = 0
            for nei in tempneis:
                count += pow(self.network.degree(nei), dilipara1)
            self.network.node[nodeindex]['setoff'] = pow(count, dilipara2)  # ?
            self.network.node[nodeindex]['nneibs'] = self.getneinum(nodeindex, samax)
        for v0, v1 in self.network.edges:
            v0index, v1index = self.edgetoindex((v0, v1))
            self.network[v0][v1]['work'] = True
            # self.network[v0][v1]['distance'] = int(pow(self.network.degree(v0)*self.network.degree(v1), distancepara))
            self.network[v0][v1]['pa'] = self.network.degree(v0)*self.network.degree(v1)
            self.network[v0][v1]['apa'] = self.network.degree(v0)+self.network.degree(v1)
            self.network[v0][v1]['lp'] = lparrays[1][v0index][v1index] + lparrays[2][v0index][v1index]*lpmult
            self.network[v0][v1]['sa'] = len(self.network.node[v0]['nneibs'] & self.network.node[v1]['nneibs'])
            self.network[v0][v1]['dw'] = dwmodels.similarity(str(v0), str(v1))  # self.get_cosin(dpmodels.get(v0), dpmodels.get(v1))
            self.network[v0][v1]['contra'] = rd.random()*self.network.number_of_nodes()

    def getneinum(self, node, deep):  # 得到某一个结点n度的邻居数
        neiset = set([node])
        for trave in range(deep):
            tempset = set()
            for tempnode in neiset:
                tempset = tempset | set(list(self.network.neighbors(tempnode)))
            neiset = neiset | tempset
        return(neiset)

    def getarrayspa(self, deep):  # 得到n跳的连接数组
        nodenum = self.network.number_of_nodes()
        arrays = np.zeros((deep, nodenum, nodenum))
        temparray = np.zeros((nodenum, nodenum))
        for v0, v1 in self.network.edges:
            v0index, v1index = self.edgetoindex((v0, v1))
            temparray[v0index][v1index] = 1
            temparray[v1index][v0index] = 1
        for trave in range(deep):
            arrays[trave] = temparray
            temparray = np.dot(temparray, arrays[0])
        return(arrays)

    def get_deepwalk(self):
        graph_adjlist = self.get_adjlist(self.network)
        dw_graph = walks.graph.from_adjlist(graph_adjlist)
        walk_length = sup.net_paras['dw_lensize']*self.network.number_of_nodes()
        walk_file = list()
        for rep in range(sup.net_paras['dw_repeat']):
            for node in self.network.nodes:
                temp_file = dw_graph.random_walk(walk_length, alpha=0.5, start=str(node))
                walk_file.append(temp_file)
        model = skipgram.Word2Vec()
        model.build_vocab(walk_file)
        model.train(walk_file, total_examples=model.corpus_count, epochs=model.iter)
        # print(model.similarity('0', '1'), model.similarity('0', '19'), model.similarity('18', '19'))
        return model

    def get_adjlist(self, net):
        graph_adjlist = list()
        for index, node in enumerate(net.nodes):
            graph_adjlist.append(list())
            neighbors = nx.neighbors(net, node)
            graph_adjlist[index].append(str(node))
            for nei in neighbors:
                graph_adjlist[index].append(str(nei))
        return graph_adjlist

    def get_cosin(self, lista, listb):
        arraya = np.array(lista).astype(np.float)
        arrayb = np.array(listb).astype(np.float)
        suma = np.sum(np.power(arraya, 2))
        sumb = np.sum(np.power(arrayb, 2))
        sum_merge = np.sum(arraya*arrayb)
        cosin = sum_merge/(math.sqrt(suma)*math.sqrt(sumb))
        return cosin

    def edgetoindex(self, etuple):  # 由于边的编号不一定对应自然数，所以需要排序后重新依据编号获取信息
        nodes = list(self.network.nodes)
        v0index = nodes.index(etuple[0])
        v1index = nodes.index(etuple[1])
        return((v0index, v1index))


if __name__ == '__main__':
    net = mynetwork(1, 20)
    net.shownet(net.network)
    pass
