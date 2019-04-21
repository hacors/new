# 处理图片，得出训练集测试集
import glob
import os
import random
import tensorflow as tf

import numpy as np
from PIL import Image

Dirorig = (r'D:\myfile\data\DMLforPRID\data\OraginData')
# Dirorig = (r'/root/data/DMLforPRID/data/OraginData')
Dirto = (r'D:\myfile\data\DMLforPRID\data\TreatedData')  # 这里对于放数据的那个文件夹的目录就好了
# Dirto = (r'/root/data/DMLforPRID/data/TreatedData')
# Dirto = (r'C:\Users\liuti\Desktop\ML_zhao\ml_code\processed')

objictsofsets = {'VIPeR': 632, 'CUHK': 1816}
dupliofsets = {'VIPeR': 2, 'CUHK': 4}
sizeofpic = 48
esize = [50, 150, 300, 500, 800]
bsize = 128
labels = [1, -2]


def allpictreat():  # 组织文件
    for DataSet in list(glob.glob(Dirorig+r'\*')):
        DataSetName = DataSet.split('\\')[-1]
        DataSetDir = Dirto+'\\'+DataSetName
        if not os.path.exists(DataSetDir):
            os.makedirs(DataSetDir)
        number_pic = {}
        for Batchs in list(glob.glob(DataSet+r'\*')):
            BatchName = Batchs.split('\\')[-1]
            for Cameras in list(glob.glob(Batchs+r'\*')):
                for PictureDir in list(glob.glob(Cameras+r'\*')):
                    PictureName = PictureDir.split('\\')[-1]
                    pic_name = BatchName[-1]+PictureName[0:3]
                    if(pic_name in number_pic):
                        number_pic[pic_name] += 1
                    else:
                        number_pic[pic_name] = 0
                    pic_name += str(number_pic[pic_name])
                    picturesplit(PictureDir, DataSetDir, pic_name)


def picturesplit(dirori, dirto, name):  # 分割并保存图片
    ima = Image.open(dirori)
    wi, hi = ima.size
    ima_cut = ima
    # ima_cut = ima.crop((wi/2-sizeofpic/2, hi/2-64, wi/2+sizeofpic/2, hi/2+64))
    ima_h = ima_cut.crop((wi/2-sizeofpic/2, 0, wi/2+sizeofpic/2, sizeofpic))
    ima_b = ima_cut.crop((wi/2-sizeofpic/2, hi/2-sizeofpic/2, wi/2+sizeofpic/2, hi/2+sizeofpic/2))
    ima_l = ima_cut.crop((wi/2-sizeofpic/2, hi-sizeofpic, wi/2+sizeofpic/2, hi))
    ima_h.save(dirto+'/'+name+'0.png')
    ima_b.save(dirto+'/'+name+'1.png')
    ima_l.save(dirto+'/'+name+'2.png')


def getsplit(setsname):  # 分割训练集和测试集
    peoplelist = []
    listlength = int(objictsofsets[setsname]/2)
    for graphnames in list(glob.glob(Dirto+'\\'+setsname+r'\*')):
        peoplename = graphnames.split('\\')[-1][0:4]
        peoplelist.append(peoplename)
    peoplelist = list(set(peoplelist))  # 去除重复，得到所有图片的编号
    random.shuffle(peoplelist)  # 所有编号打乱
    trainselect = peoplelist[0:listlength]
    testselect = peoplelist[listlength:listlength*2]
    return trainselect, testselect


def listTOtuple(setsname, picselect):
    temptuple = []
    tempsimbol = []
    tempindex = []
    for tra_a in picselect:  # 得到正负序列
        temptuple.append([tra_a, tra_a])
        samenum = True
        while(samenum):
            another = picselect[random.randint(0, len(picselect)-1)]
            if(not tra_a == another):
                samenum = False
        temptuple.append([tra_a, another])
    for index, tra_a in enumerate(temptuple):
        tempindex.append(index)
        if tra_a[0] == tra_a[1]:
            tempsimbol.append(labels[0])
        else:
            tempsimbol.append(labels[1])
        tra_a[0] += str(random.randint(0, dupliofsets[setsname]/2-1))
        tra_a[1] += str(random.randint(dupliofsets[setsname]/2, dupliofsets[setsname]-1))
    return temptuple, tempsimbol, tempindex  # 返回训练集序列和测试集编号


def listTOcmc(setsname, picselect):
    temptuple = []
    tempsimbol = []
    tempindex = []
    for indexb, tra in enumerate(picselect):
        temptuple.append([tra+str(0), tra + str(1)])
        tempsimbol.append(1)
        tempindex.append(indexb)
    return temptuple, tempsimbol, tempindex


def tupletotensor(setsname, tupledata, simboldata, indexdata):
    list_data = []
    array_simbol = np.array(simboldata)
    array_index = np.array(indexdata)
    for tra_a in range(len(indexdata)):
        list_data.append([])
        for tra_b in range(3):
            list_data[tra_a].append([])
            for tra_c in range(2):
                pic_name = tupledata[tra_a][tra_c]+str(tra_b)
                dirtopic = Dirto+'\\'+setsname+'\\'+pic_name+'.png'
                pic = Image.open(dirtopic)
                data = (np.array(pic, dtype=np.float)/255).tolist()
                list_data[tra_a][tra_b].append(data)
    array_data = np.array(list_data)
    tensor_data = tf.data.Dataset.from_tensor_slices((array_data, array_simbol, array_index))
    return tensor_data


def getPictures(setsname):
    trainselect, testselect = getsplit(setsname)
    traintuple, trainsimbol, trainindex = listTOtuple(setsname, trainselect)
    testtuple, testsimbol, testindex = listTOtuple(setsname, testselect)
    covdatatrain = tupletotensor(setsname, traintuple, trainsimbol, trainindex)
    covdatatrain = covdatatrain.shuffle(buffer_size=1000).repeat(esize).batch(bsize)
    covdatatest = tupletotensor(setsname, testtuple, testsimbol, testindex)
    covdatatest = covdatatest.batch(bsize)
    return covdatatrain, covdatatest


def getcmc(setsname):
    trainselect, testselect = getsplit(setsname)
    cmctuple, cmcsimbol, cmcindex = listTOcmc(setsname, testselect)
    return cmctuple, cmcindex, cmcindex


def showima(array):  # 输出图片
    array = array*255.0
    array = array.astype(np.uint8)
    img = Image.fromarray(array, 'RGB')
    img.show()


# test
if __name__ == '__main__':
    traindata, testdata = getPictures('VIPeR')
    iterator = traindata.make_one_shot_iterator()
    batch_traindata, batch_trainlabels, batch_trainindex = iterator.get_next()
    with tf.train.MonitoredSession() as sess:
        while not sess.should_stop():
            outdata, outlabels, outindex = sess.run([batch_traindata, batch_trainlabels, batch_trainindex])
            print(outdata, outlabels, outindex)
            for tra_a in outdata:
                for tra_b in tra_a:
                    for tra_c in tra_b:
                        myarray = np.array(tra_c)
                        showima(myarray)
