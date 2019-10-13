# 配置全局变量

import os
import shutil


DIR = 'Datasets/Xiamen_data/'
DATASET_NAMES = ['train', 'test', 'submit']


# 原始数据地址
origin_dir = DIR+'origin/'

origin_train_data_path = origin_dir+'train.csv'
origin_train_target_path = origin_dir+'train_target.csv'
origin_test_data_path = origin_dir+'test.csv'

# 划分了训练集、测试集的地址，可以按照某种地区提取特征，置于feature文件夹中
# 同时需要给出预测集的处理地址
split_dir = DIR+'split/'

train_dir = split_dir+'train/'
train_feature_dir = split_dir+'train/feature/'
train_data = train_dir+'data.csv'
train_feed = train_dir+'feed.csv'
train_target = train_dir+'target.csv'

test_dir = split_dir+'test/'
test_feature_dir = split_dir+'test/feature/'
test_data = test_dir+'data.csv'
test_feed = test_dir+'feed.csv'
test_target = test_dir+'target.csv'

submit_dir = split_dir+'submit/'
submit_feature_dir = split_dir+'submit/feature/'
submit_data = submit_dir+'data.csv'
submit_feed = submit_dir+'feed.csv'

result_dir = DIR+'result/'
model_dir = DIR+'model/'

# 其他数据
SPLIT_TEST_TRAIN = 47833


# 定义生成文件夹函数
def mkdirector(director):
    director = director[:-1]  # 去除最后一个斜杠
    if os.path.exists(director):
        shutil.rmtree(director)
    else:
        os.mkdir(director)


if __name__ == '__main__':
    pass
