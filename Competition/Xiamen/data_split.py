# 分割数据集，将数据集存储于设置的位置

import pandas as pd
import config

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
    train_data = origin_train_data[config.SPLIT_TEST_TRAIN:]
    train_target = origin_train_target[config.SPLIT_TEST_TRAIN:]
    train_data.to_csv(config.train_dir+'/data.csv', index=False)
    train_target.to_csv(config.train_dir+'/target.csv', index=False)

    test_data = origin_train_data[config.SPLIT_TEST_TRAIN:]
    test_target = origin_train_target[config.SPLIT_TEST_TRAIN:]
    test_data.to_csv(config.test_dir+'/data.csv', index=False)
    test_target.to_csv(config.test_dir+'/target.csv', index=False)

    submit_data = origin_test_data
    submit_data.to_csv(config.submit_dir+'/data.csv', index=False)
