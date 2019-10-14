# 分割数据集，将数据集存储于设置的位置

import pandas as pd
import config
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    origin_train_data = pd.read_csv(config.origin_train_data_path)
    origin_train_target = pd.read_csv(config.origin_train_target_path).drop(['id'], axis=1)  # 删除无关数据
    origin_test_data = pd.read_csv(config.origin_test_data_path)

    # 创建分割数据集需要的所有文件夹
    config.mkdirector(config.split_dir)
    config.mkdirector(config.result_dir)
    config.mkdirector(config.model_dir)

    config.mkdirector(config.train_dir)
    config.mkdirector(config.test_dir)
    config.mkdirector(config.submit_dir)

    # 分割并存储数据
    '''
    # 由于数据分布有问题，采用随机抽样方法
    train_data = origin_train_data[:3000]
    train_target = origin_train_target[0:3000]
    train_data.to_csv(config.train_dir+'/data.csv', index=False)
    train_target.to_csv(config.train_dir+'/target.csv', index=False)

    test_data = origin_train_data[3000:config.SPLIT_TEST_TRAIN]
    test_target = origin_train_target[3000:config.SPLIT_TEST_TRAIN]
    test_data.to_csv(config.test_dir+'/data.csv', index=False)
    test_target.to_csv(config.test_dir+'/target.csv', index=False)
    '''

    train_data, test_data, train_target, test_target = train_test_split(origin_train_data, origin_train_target, test_size=0.25, shuffle=True)
    train_data.to_csv(config.train_dir+'/data.csv', index=False)
    train_target.to_csv(config.train_dir+'/target.csv', index=False)
    test_data.to_csv(config.test_dir+'/data.csv', index=False)
    test_target.to_csv(config.test_dir+'/target.csv', index=False)

    submit_data = origin_test_data
    submit_data.to_csv(config.submit_dir+'/data.csv', index=False)
