import multiprocessing as mp
import random as rd

import numpy as np
from matplotlib import pyplot as plt

import movenet
import support as sup

data_director = 'LINKRE/python/temp/data.txt'
graph_director = 'LINKRE/python/temp/'


class experiment():
    def __init__(self, gratp=1, breaknum=20, nodenum=10, epoch=30, generate=1.2, repeat=50):
        # def __init__(self, gratp=1, breaknum=20, nodenum=10, epoch=30, generate=1.2, repeat=50):
        self.gratp = gratp
        self.breaknum = breaknum
        self.nodenum = nodenum
        self.epoch = epoch
        self.generate = generate
        self.repeat = repeat

    def singleexp(self, index):
        print('run', index)
        net = movenet.movenet(self.gratp, self.breaknum, nodenum=self.nodenum)
        alllist = list()
        rec_num = len(sup.rec_types)
        for recindex in range(1, rec_num+1):
            recstr = str(sup.rec_types(recindex)).split('.')[-1]
            templist = net.recovery(recstr, self.epoch, self.generate)
            alllist.append(templist)
        ranks = self.give_rank(alllist)
        return ranks

    def give_rank(self, alllist):
        temp_array = np.array(alllist)
        rank_array = np.argsort(temp_array, axis=0)
        rank_array = np.argsort(rank_array, axis=0)
        rank_array = (len(sup.rec_types)-rank_array)/len(sup.rec_types)  # 将index与数据绑定
        rank_array = rank_array.flatten()
        return rank_array

    def allexp(self):
        pool_process = mp.pool.Pool()
        result = pool_process.map(self.singleexp, range(self.repeat))
        np.savetxt(data_director, result)

    def read_show(self, name):
        result = np.zeros((len(sup.rec_types), self.breaknum+1))
        datas = np.loadtxt(data_director)
        for data in datas:
            stru_data = data.reshape((len(sup.rec_types), self.breaknum+1))
            result = result+stru_data
        self.simpledraw(result, name)

    def simpledraw(self, thelist, name='default'):
        plt.figure(figsize=(19, 12))
        title_name = name+'(gratp:%s breaknum:%s nodenum:%s repeat:%s)' % (sup.net_types(self.gratp), self.breaknum, self.nodenum, self.repeat)
        plt.title(title_name)
        reclist = range(self.breaknum+1)  # 需要0恢复完成所有list，所以+1
        for recindex, trave in enumerate(thelist):
            recstr = str(sup.rec_types(recindex+1)).split('.')[-1]
            if recstr == 'contra':
                plt.plot(reclist, trave, color=[0.0, 0.0, 0.0, 1.0], label=recstr)
            elif recstr[0] == 'r':
                plt.plot(reclist, trave, color=[rd.random(), rd.random(), rd.random(), 0.7], label=recstr, linestyle='--', marker='s')
            else:
                plt.plot(reclist, trave, color=[rd.random(), rd.random(), rd.random(), 0.7], label=recstr, linestyle='-', marker='d')
        plt.legend()
        plt.savefig(graph_director+name+'.png')


if __name__ == '__main__':
    myexp = experiment()
    myexp.allexp()
    myexp.read_show('merged_graph')
