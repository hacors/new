import time
import multiprocessing as multp

import numpy as np
from matplotlib import pyplot as plt

import movenet
import support as sup

gratp = 1  # 实验网络的类型
breaknum = 60  # 破坏的边的数目
nodenum = 40  # 结点数目
epoch = 50  # 最终粒子计数时产生的代数
repeat = 100  # 实验重复次数
generate = 1.5  # 例子产生的效率
rec_num = len(sup.rec_types)
gra_name = str(sup.net_types(gratp)).split('.')[-1]
INFO = 'type_%s bnum_%s nnum_%s epoch_%s repeat_%s' % (gra_name, breaknum, nodenum, epoch, repeat)
data_director = sup.ROOT + '/temp/data %s.npy' % INFO
merge_director = sup.ROOT + '/temp/merge %s.png' % INFO
single_director = sup.ROOT + '/temp/single %s.png' % INFO


def get_rank(index):  # 单次实验，通过比对所有的方法，在同一个网络的环境下恢复过程中的排序计分情况
    strtime = time.strftime("%H:%M %m-%d", time.localtime())
    print('run %s at time(%s)' % (index, strtime))
    net = movenet.movenet(gratp, breaknum, nodenum)
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


def get_process_data(index):  # 获取单次实验的完整过程数据
    net = movenet.movenet(gratp, breaknum, nodenum)
    alllist = list()
    for recindex in range(1, rec_num+1):
        recstr = str(sup.rec_types(recindex)).split('.')[-1]
        templist = net.recovery(recstr, epoch, generate)
        alllist.append(templist)
    return alllist


def read_datas(director):
    result = np.zeros((rec_num, breaknum+1))
    datas = np.load(director)
    for data in datas:
        stru_data = data.reshape((rec_num, breaknum+1))
        result = result+stru_data
    return result


def draw(the_list, graph_dir):
    plt.figure(figsize=(19, 12))
    plt.title(INFO)
    reclist = range(breaknum+1)  # 需要从0恢复完成所有list，所以+1
    temp = int(rec_num/2)
    for recindex in range(temp):
        temp_color = sup.colors[recindex]
        indexa, indexb = recindex, recindex+temp+1
        plt.plot(reclist, the_list[indexa], color=temp_color, label=sup.rec_types(indexa+1), linestyle='-', marker='d')
        plt.plot(reclist, the_list[indexb], color=temp_color, label=sup.rec_types(indexb+1), linestyle='--', marker='s')
    plt.plot(reclist, the_list[temp], color=[0.0, 0.0, 0.0, 1.0], label=sup.rec_types(temp+1))
    plt.legend()
    plt.savefig(graph_dir)


def single_experiment():
    temp_recover_list = get_process_data(0)
    draw(temp_recover_list, single_director)


if __name__ == '__main__':
    pool_result = list()
    pool = multp.Pool(processes=10)
    for index in range(repeat):
        pool_result.append(pool.apply_async(get_rank, (index, )))
    pool.close()
    pool.join()
    true_result = list()
    for temp in pool_result:
        true_result.append(temp.get())
    np.save(data_director, np.array(true_result))
    merged_result = read_datas(data_director)
    draw(merged_result, merge_director)
