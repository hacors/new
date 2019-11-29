'''
设置全局的配置文件，比如下载地址，运行环境等等
'''
import os
DATA_ROOT = os.path.join(os.getcwd(), 'Data_test')
os.mkdir(DATA_ROOT)
DIR_LIST = {'ess': 'Data'}
