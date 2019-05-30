import random as rd
import time
import multiprocessing as multp

import numpy as np
from matplotlib import pyplot as plt

import movenet
import support as sup

data_director = 'LINKRE/python/temp/data'
graph_director = 'LINKRE/python/temp/'

'''
gratp = 1
breaknum = 30
nodenum = 15
epoch = 30
generate = 1.2
repeat = 100
'''
gratp = 1
breaknum = 30
nodenum = 15
epoch = 30
generate = 1.5
repeat = 20

net = movenet.movenet(gratp, breaknum, nodenum=nodenum)
rec_num = len(sup.rec_types)
gra_name = str(sup.net_types(gratp)).split('.')[-1]


def singleexp(index):
    strtime = time.strftime("%H:%M %m-%d", time.localtime())
    print('run %s at time(%s)' % (index, strtime))
    alllist = list()
    for recindex in range(1, rec_num+1):
        recstr = str(sup.rec_types(recindex)).split('.')[-1]
        templist = net.recovery(recstr, epoch, generate)
        alllist.append(templist)
    temp_array = np.array(alllist)
    rank_array = np.argsort(temp_array, axis=0)
    rank_array = np.argsort(rank_array, axis=0)
    rank_array = (rec_num-rank_array)/rec_num  # 将index与数据绑定
    rank_array = rank_array.flatten()
    return rank_array


def read_show(director, name):
    result = np.zeros((rec_num, breaknum+1))
    datas = np.loadtxt(director)
    for data in datas:
        stru_data = data.reshape((rec_num, breaknum+1))
        result = result+stru_data
    simpledraw(result, name)


def simpledraw(thelist, name='default'):
    plt.figure(figsize=(19, 12))
    title_name = name+' (gratps:%s breaknum:%s nodenum:%s repeat:%s)' % (gra_name, breaknum, nodenum, repeat)
    plt.title(title_name)
    reclist = range(breaknum+1)  # 需要从0恢复完成所有list，所以+1
    temp = int(rec_num/2)
    for recindex in range(temp):
        temp_color = list((rd.random(), rd.random(), rd.random(), 0.7))
        indexa, indexb = recindex, recindex+temp+1
        plt.plot(reclist, thelist[indexa], color=temp_color, label=sup.rec_types(indexa+1), linestyle='-', marker='d')
        plt.plot(reclist, thelist[indexb], color=temp_color, label=sup.rec_types(indexb+1), linestyle='--', marker='s')
    plt.plot(reclist, thelist[temp], color=[0.0, 0.0, 0.0, 1.0], label=sup.rec_types(temp+1))
    plt.legend()
    plt.savefig(graph_director+name+'.png')


if __name__ == '__main__':
    all_result = list()
    pool_process = multp.Pool(processes=10)
    for index in range(repeat):
        all_result.append(pool_process.apply_async(singleexp, (index,)))
    np.savetxt(data_director, all_result)
    read_show(data_director, 'merged_graph')
