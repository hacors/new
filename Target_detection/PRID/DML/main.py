import graph_slice as gs
import net_design as nd
import tensorflow as tf
import digshow as ds
import numpy as np


setsname = 'VIPeR'
Graphsave = r'D:\myfile\data\DMLforPRID\model\logs'
# Graphsave = r'/root/data/DMLforPRID/model/logs'
Modelsave = r'D:\myfile\data\DMLforPRID\model'
# Modelsave = r'/root/data/DMLforPRID/model/trainedmodel'
tf.device('/gpu:1')


class beginrun():
    def __init__(self, name):
        self.trip_scnn = nd.triple_scnn([[7, 7, 3, 32], [5, 5, 32, 48]], [[6912, 500]],
                                        [None, 48, 48, 3], setsname, True)
        self.train_data, self.test_data = gs.getPictures(setsname)
        self.sets_name = name

    def train(self, model_choose):
        batchnum = int(gs.objictsofsets[self.sets_name]*model_choose/gs.bsize)
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.trip_scnn.losssums, var_list=tf.trainable_variables())
        init = tf.global_variables_initializer()
        iterator = self.train_data.make_one_shot_iterator()
        bt_train_data, bt_train_labels, bt_train_index = iterator.get_next()
        saver = tf.train.Saver()
        batch_index = []
        all_loss = []
        with tf.Session() as sess_train:
            sess_train.run(init)
            for tra in range(batchnum):
                true_train_data, true_train_labels, true_train_index = sess_train.run(
                    [bt_train_data, bt_train_labels, bt_train_index])
                all_feed_dict = {self.trip_scnn.all_data: true_train_data, self.trip_scnn.labels: true_train_labels}
                sess_train.run(train_step, feed_dict=all_feed_dict)
                loss = sess_train.run(self.trip_scnn.losssums, feed_dict=all_feed_dict)
                if(tra % 20 == 0):
                    batch_index.append(tra)
                    all_loss.append(loss)
                print(batchnum, ' ', tra)
            saver.save(sess_train, Modelsave+'\\%sepoch\\%s_model.ckpt' % (model_choose, self.sets_name))
        print(batch_index)
        print(all_loss)
        ds.scatter(batch_index, all_loss)

    def simpletest(self, model_choose, datas, batchnum):
        saver = tf.train.Saver()
        iterator = datas.make_one_shot_iterator()
        bt_train_data, bt_train_labels, bt_train_index = iterator.get_next()
        allcosin = [[], [], [], []]
        alllabel = []
        with tf.Session() as sess_test:
            saver.restore(sess_test, Modelsave+'\\%sepoch\\%s_model.ckpt' % (model_choose, self.sets_name))
            for tra in range(batchnum):
                true_test_data, true_test_labels, true_test_index = sess_test.run([bt_train_data, bt_train_labels, bt_train_index])
                all_feed_dict = {self.trip_scnn.all_data: true_test_data, self.trip_scnn.labels: true_test_labels}
                cosin0, cosin1, cosin2, cosinsum = sess_test.run([self.trip_scnn.scnn_0.cosin, self.trip_scnn.scnn_1.cosin,
                                                                  self.trip_scnn.scnn_2.cosin, self.trip_scnn.cosinsums], feed_dict=all_feed_dict)
                allcosin[0].extend(cosin0)
                allcosin[1].extend(cosin1)
                allcosin[2].extend(cosin2)
                allcosin[3].extend(cosinsum)
                alllabel.extend(true_test_labels)
        return(allcosin, alllabel)
        '''
        tflabels = self.changedata(alllabel)
        rates = []
        length = len(tflabels)
        for tra in allcosin:
            tfcosin = self.changedata(tra)
            same = list(map(lambda x, y: x == y, tflabels, tfcosin))
            trues = same.count(True)
            rates.append(trues/length)
        return rates
        '''

    def cmctest(self):
        saver = tf.train.Saver()
        trainselect, testselect = gs.getsplit(setsname)
        tuples, simbols, indexs = gs.listTOcmc(setsname, testselect)
        lists = [[[], []], [[], []], [[], []]]
        with tf.Session() as sess_test:
            saver.restore(sess_test, Modelsave+'\\%sepoch\\%s_model.ckpt' % (800, self.sets_name))
            tempcovdata = gs.tupletotensor(setsname, tuples, simbols, indexs)
            tempcovdata = tempcovdata.batch(gs.bsize)
            batchnum = int(len(indexs)/gs.bsize)+1
            iterator = tempcovdata.make_one_shot_iterator()
            bt_train_data, bt_train_labels, bt_train_index = iterator.get_next()
            for trb in range(batchnum):
                true_test_data, true_test_labels, true_test_index = sess_test.run([bt_train_data, bt_train_labels, bt_train_index])
                all_feed_dict = {self.trip_scnn.all_data: true_test_data, self.trip_scnn.labels: true_test_labels}
                out00, out01 = sess_test.run([self.trip_scnn.scnn_0.cnn_0.output, self.trip_scnn.scnn_0.cnn_1.output], feed_dict=all_feed_dict)
                out10, out11 = sess_test.run([self.trip_scnn.scnn_1.cnn_0.output, self.trip_scnn.scnn_1.cnn_1.output], feed_dict=all_feed_dict)
                out20, out21 = sess_test.run([self.trip_scnn.scnn_2.cnn_0.output, self.trip_scnn.scnn_2.cnn_1.output], feed_dict=all_feed_dict)
                lists[0][0].extend(out00.tolist())
                lists[0][1].extend(out01.tolist())
                lists[1][0].extend(out10.tolist())
                lists[1][1].extend(out11.tolist())
                lists[2][0].extend(out20.tolist())
                lists[2][1].extend(out21.tolist())
        return(lists)

    def getcovpic(self):
        saver = tf.train.Saver()
        lists = [[], [], []]
        with tf.Session() as sess_test:
            saver.restore(sess_test, Modelsave+'\\%sepoch\\%s_model.ckpt' % (800, self.sets_name))
            out0 = sess_test.run(self.trip_scnn.scnn_0.cnn_0.conv_wbs)
            out1 = sess_test.run(self.trip_scnn.scnn_1.cnn_0.conv_wbs)
            out2 = sess_test.run(self.trip_scnn.scnn_2.cnn_0.conv_wbs)
            lists[0].extend(out0.tolist())
            lists[1].extend(out1.tolist())
            lists[2].extend(out2.tolist())
        return(lists)

    def showcmc(self, alldatas):
        allcosins = [[], [], [], []]
        alltops = [[], [], [], []]
        sums = [[], [], [], []]
        pcindexs = []
        lenght = len(alldatas[0][0])
        for tra_a in range(lenght):
            pcindexs.append(tra_a)
        for indexa, tra_a in enumerate(alldatas):
            temp0 = tra_a[0]
            temp1 = tra_a[1]
            for indexb, tra_b in enumerate(temp0):
                allcosins[indexa].append([])
                for tra_c in temp1:
                    tempcosin = self.getcosin(tra_b, tra_c)
                    allcosins[indexa][indexb].append(tempcosin)
        allcosins[3] = (np.array(allcosins[0])+np.array(allcosins[1])+np.array(allcosins[2])).tolist()
        for indexa, tra_a in enumerate(allcosins):
            for indexb, tra_b in enumerate(tra_a):
                tops = self.gettop(tra_b, indexb)
                alltops[indexa].append(tops)
        for indexa, tra_a in enumerate(alltops):
            tempsum = 0
            for indexb, tra_b in enumerate(tra_a):
                tempsum += tra_a.count(indexb)
                sums[indexa].append(tempsum/lenght)
        ds.scatter(pcindexs, sums[0], color='r')
        ds.scatter(pcindexs, sums[1], color='b')
        ds.scatter(pcindexs, sums[2], color='g')
        ds.scatter(pcindexs, sums[3], color='black')

    def getcosin(self, lista, listb):
        arraya = np.array(lista)
        arrayb = np.array(listb)
        arrayax = np.sqrt(np.sum(np.multiply(arraya, arraya)))
        arraybx = np.sqrt(np.sum(np.multiply(arrayb, arrayb)))
        arrayab = np.sum(np.multiply(arraya, arrayb))
        return(arrayab/(arrayax*arraybx))

    def changedata(self, inputlist):
        outlist = []
        for tra in inputlist:
            if tra > 0:
                outlist.append(True)
            else:
                outlist.append(False)
        return outlist

    def gettop(self, listname, index):
        a = listname[index]
        num = 0
        for tra in listname:
            if tra >= a:
                num += 1
        return(num)


if __name__ == '__main__':
    myexperiment = beginrun('VIPeR')
    '''
    alldatas = myexperiment.cmctest()
    myexperiment.showcmc(alldatas)
    '''
    myexperiment.getcovpic()
# tensorboard --logdir D:\myfile\data\DMLforPRID\model\logs --port 6006
