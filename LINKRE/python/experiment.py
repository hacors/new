import random as rd

import numpy as np
from matplotlib import pyplot as plt

import movenet
import support as sup


class experiment():
    def __init__(self, gratp=1, breaknum=20, nodenum=10, epoch=50, generate=1.2, repeat=50):
        self.net = movenet.movenet(gratp, breaknum, nodenum=nodenum)
        self.breaknum = breaknum
        self.epoch = epoch
        self.generate = generate
        self.repeat = repeat

    def singleexp(self):
        alllist = list()
        rec_num = len(sup.rec_types)
        for recindex in range(1, rec_num+1):
            recstr = str(sup.rec_types(recindex)).split('.')[-1]
            templist = self.net.recovery(recstr, self.epoch, self.generate)
            alllist.append(templist)
        return alllist

    def allexp(self):
        result = np.zeros((len(sup.rec_types), self.breaknum+1))
        for index in range(self.repeat):
            temp_list = self.singleexp()
            temp_array = np.array(temp_list)
            smallest_index = temp_array.argmin(axis=0)
            map()

    def simpledraw(self, thelist, name='default'):
        plt.figure()
        plt.title(name)
        reclist = range(self.breaknum+1)  # 需要0恢复完成所有list，所以+1
        for recindex, trave in enumerate(thelist):
            recstr = str(sup.rec_types(recindex)).split('.')[-1]
            if recstr == 'contra':
                plt.plot(reclist, trave, color=[0.0, 0.0, 0.0, 1.0], label=recstr)
            elif recstr[0] == 'r':
                plt.plot(reclist, trave, color=[rd.random(), rd.random(), rd.random(), 0.7], label=recstr, linestyle='--', marker='s')
            else:
                plt.plot(reclist, trave, color=[rd.random(), rd.random(), rd.random(), 0.7], label=recstr, linestyle='-', marker='d')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    myexp = experiment()
    myexp.allexp()
'''    alllist = myexp.singleexp()
    myexp.simpledraw(alllist)'''
