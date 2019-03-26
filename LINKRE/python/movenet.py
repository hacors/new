import math
import random as rd

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import support as sup
import naivenet

# from multiprocessing import process


class movenet(naivenet.mynetwork):
    '''
    gratps:图的类型,
    breaknum:破坏边的数目,
    nodenum:结点数目,
    '''

    def __init__(self, gratps, breaknum, nodenum=None):
        super().__init__(gratps, nodenum)
        self.breaklist = self.takebreak(breaknum)

    @sup.log
    def takebreak(self, breaknum):
        breaklist = list()
        alllinks = list(self.network.edges)
        linknum = len(alllinks)
        if(breaknum > linknum):
            breaknum = linknum
        breakindex = np.random.choice(linknum, breaknum, replace=False)
        for trave in breakindex:
            bedge = alllinks[trave]
            breaklist.append(bedge)
            self.network[bedge[0]][bedge[1]]['work'] = False
        return(breaklist)

    @sup.log
    def getpath(self):  # 通过这一个做控制
        tempnet = self.network.copy()
        for v0, v1 in self.network.edges:
            if not self.network[v0][v1]['work']:
                tempnet.remove_edge(v0, v1)
        for source in tempnet.nodes:
            # temp_path = nx.single_source_dijkstra_path(tempnet, source, weight='distance')
            temp_path = nx.single_source_shortest_path(tempnet, source)
            temp_dict = dict()
            for target in temp_path:
                if not source == target:
                    temp_dict[target] = temp_path[target][1]
            self.network.node[source]['choice'] = temp_dict

    @sup.log
    def takemove(self, epoch, generate):
        '''
        epoch:代数,
        generate:粒子生成比率,
        '''
        sumlist = list()
        sumofpa = 0
        allnodes = list(self.network.nodes)
        for thenode in self.network.nodes:
            self.network.node[thenode]['atwait'] = list()
            self.network.node[thenode]['arriver'] = list()
            self.network.node[thenode]['cantgo'] = list()
        for edgev1, edgev2 in self.network.edges:
            self.network[edgev1][edgev2]['ways'] = list()
        for epo in range(epoch):
            for index, thenode in enumerate(self.network.nodes):
                # 生成粒子
                genenum = math.floor(generate)
                if(rd.random() <= generate-genenum):
                    genenum += 1
                sumofpa += genenum
                candidate = allnodes.copy()
                candidate.pop(index)
                choosed = list(np.random.choice(candidate, genenum, replace=True))  # 目标可以重复
                alltomove = self.network.node[thenode]['atwait'] + \
                    self.network.node[thenode]['arriver']+choosed
                self.network.node[thenode]['atwait'].clear()
                self.network.node[thenode]['arriver'].clear()
                # 开始运动
                setoff = self.network.node[thenode]['setoff']
                numhasgo = 0
                for destination in alltomove:
                    if destination in self.network.node[thenode]['choice']:  # 可移动
                        if numhasgo < setoff:  # 还有剩余发车量
                            nextjump = self.network.node[thenode]['choice'][destination]
                            if nextjump == destination:  # 到站粒子消失
                                sumofpa -= 1
                            else:
                                self.network.node[nextjump]['arriver'].append(destination)
                            numhasgo += 1
                        else:  # 发车饱和
                            self.network.node[thenode]['atwait'].append(destination)
                    else:  # 不可移动
                        self.network.node[thenode]['cantgo'].append(destination)
            sumlist.append(sumofpa)
        return(sumlist)

    @sup.log
    def showbadnet(self, network, name='default'):
        plt.cla()  # 清除画布
        pos = nx.kamada_kawai_layout(network)
        edge_work = [(v0, v1) for (v0, v1) in network.edges() if (network[v0][v1]['work'])]
        edge_break = [(v0, v1) for (v0, v1) in network.edges() if not (network[v0][v1]['work'])]
        nx.draw_networkx_edges(network, pos, edgelist=edge_work, edge_color='black', width=1.0)  # 画出有效边
        nx.draw_networkx_edges(network, pos, edgelist=edge_break, edge_color='red', width=1.0,  style='dashed')  # 画出损坏边
        nx.draw_networkx_nodes(network, pos, node_color='green', node_size=300)  # 画出结点
        nx.draw_networkx_labels(network, pos, font_size=8, font_color='white')
        direct = 'LINKRE/python/graphs/'+name+'.png'  # 保存图片路径
        plt.savefig(direct)
        plt.close()  # 关闭

    @sup.log
    def recovery(self, recstr, epoch, generate):
        '''
        recstr:链路权重的选择,
        epoch:测试网络时粒子生成的代数,
        generate:粒子生成的几率
        '''
        templist = self.breaklist.copy()
        if(recstr[0] == 'r'):
            tempstr = recstr[1:]
            templist.sort(key=lambda x: (self.network[x[0]][x[1]][tempstr]))
        else:
            templist.sort(key=lambda x: (self.network[x[0]][x[1]][recstr]))
            templist = templist[::-1]
        # self.showbadnet(self.network, recstr+'0')
        self.getpath()
        partnumsum = self.takemove(epoch, generate)[-1]
        print(partnumsum, 'have rec 0')
        sumlist = list()
        sumlist.append(partnumsum)
        while(templist):  # 每恢复一条计算一次
            recovering = templist[0]
            self.network[recovering[0]][recovering[1]]['work'] = True
            templist.pop(0)
            recovednum = len(self.breaklist)-len(templist)
            # self.showbadnet(self.network, recstr+str(recovednum))
            self.getpath()
            partnumsum = self.takemove(epoch, generate)[-1]
            print(partnumsum, 'have rec %s' % recovednum)
            sumlist.append(partnumsum)
        for v0, v1 in self.breaklist:
            self.network[v0][v1]['work'] = False  # 恢复为开始的状态
        return(sumlist)


if __name__ == '__main__':
    pass
