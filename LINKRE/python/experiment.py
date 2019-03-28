import multiprocessing as mp
import random as rd

import numpy as np
from matplotlib import pyplot as plt

import movenet
import support as sup

data_director = 'LINKRE/python/temp/data.txt'
graph_director = 'LINKRE/python/temp/'


class experiment():
    def __init__(self, gratp=1, breaknum=40, nodenum=20, epoch=80, generate=1.2, repeat=200):
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
        return alllist

    def allexp(self):
        pool_process = mp.pool.Pool()
        results = pool_process.map(self.singleexp, range(self.repeat))
        np.savetxt(data_director, results)
        '''
        # result = np.zeros((len(sup.rec_types), self.breaknum+1))
        for repe in range(self.repeat):
            process = mp.process.BaseProcess(target=self.singleexp(), args=(self.result,))
            temp_list = self.singleexp()
            temp_array = np.array(temp_list)
            smallest_index = temp_array.argmin(axis=0)
            for index, choose in enumerate(smallest_index):
                result[choose][index] += 1
            print(repe)
            np.savetxt(data_director, result)
        '''

    def read_show(self, name):
        datas = np.loadtxt(data_director)
        datas = datas.reshape((len(sup.rec_types), self.breaknum+1))
        self.simpledraw(datas, name)

    def simpledraw(self, thelist, name='default'):
        plt.figure(figsize=(19, 12))
        plt.title(name)
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
    # myexp.read_show('merged_graph')
