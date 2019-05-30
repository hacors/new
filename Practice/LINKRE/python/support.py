from enum import Enum
import time
net_types = Enum('type', ('ba', 'usairport', 'top500usairport', 'airtraffic', 'powergrid', 'openflight', 'test'))
rec_types = Enum('type', ('pa', 'apa', 'lp', 'sa', 'dw', 'contra', 'rpa', 'rapa', 'rlp', 'rsa', 'rdw'))
net_paras = {'balinks': 3}
rec_paras = {'lpmax': 3, 'lpmult': 0.4, 'samax': 2}
mov_paras = {'dilipara1': 1, 'dilipara2': 1}  # dilipara为结点最大出发量的系数
ROOT = 'Practice/LINKRE/python'


def log(func):
    def wrapper(*args, **kw):
        print('----------------call (%s) at time(%s)' % (func.__name__, time.ctime()))
        return(func(*args, **kw))
    return(wrapper)


if __name__ == '__main__':
    pass
