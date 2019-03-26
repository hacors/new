import movenet
import support as sup
from matplotlib import pyplot as plt


class experiment():
    def __init__(self, gratp=1, breaknum=20, nodenum=10, epoch=40, generate=0.8, repeat=10):
        self.net = movenet.movenet(gratp, breaknum, nodenum=nodenum)
        self.epoch = epoch
        self.generate = generate
        self.repeat = repeat

    def singleexp(self):
        alllist = list()
        for recindex in range(11):
            recindex = recindex+1
            recstr = str(sup.rec_types(recindex)).split('.')[-1]
            templist = self.net.recovery(recstr, self.epoch, self.generate)
            alllist.append([templist, recstr])
        return alllist

    def allexp(self):
        pass

    def simpledraw(self, thelist, name='default'):
        plt.figure()
        plt.title(name)
        reclist = range(len(self.net.breaklist)+1)  # 需要0~所有list，所以+1
        for trave in thelist:
            '''
            if trave[1] == 'contra':
                plt.plot(reclist, trave[0], color='black', label=trave[1])
            elif trave[1] == 'pa':
                plt.plot(reclist, trave[0], color='blue', label=trave[1], marker='v')
            elif trave[1] == 'apa':
                plt.plot(reclist, trave[0], color='blue', label=trave[1], marker='1')
            elif trave[1] == 'lp':
                plt.plot(reclist, trave[0], color='blue', label=trave[1], marker='2')
            elif trave[1] == 'sa':
                plt.plot(reclist, trave[0], color='blue', label=trave[1], marker='3')
            elif trave[1] == 'dw':
                plt.plot(reclist, trave[0], color='blue', label=trave[1], marker='4')
            elif trave[1] == 'rpa':
                plt.plot(reclist, trave[0], color='red', label=trave[1], marker='v')
            elif trave[1] == 'rapa':
                plt.plot(reclist, trave[0], color='red', label=trave[1], marker='1')
            elif trave[1] == 'rlp':
                plt.plot(reclist, trave[0], color='red', label=trave[1], marker='2')
            elif trave[1] == 'rsa':
                plt.plot(reclist, trave[0], color='red', label=trave[1], marker='3')
            elif trave[1] == 'rdw':
                plt.plot(reclist, trave[0], color='red', label=trave[1], marker='4')
            else:
                pass
            '''
        plt.legend()
        plt.show()


if __name__ == '__main__':
    myexp = experiment()
    alllist = myexp.singleexp()
    # alllist = [[[1, 2], 'rpa'], [[3, 3], 'pa']]
    myexp.simpledraw(alllist)
