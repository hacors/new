import random as rd
import time

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import config

refer_columns = ['certId', 'dist', 'bankCard', 'residentAddr']


# 数据预处理并返回处理好的数据
def preprocess(origin_data):
    preprocessed_data = origin_data
    preprocessed_data = preprocessed_data.drop(['id'], axis=1)  # 去除无关变量
    preprocessed_data = preprocessed_data.fillna(-999).astype(np.float64).replace(-999.0, -1.0).astype(str)  # 调整所有格式

    # 转换时间戳
    def get_begin_time(begin_str):
        timeArray = time.localtime(int(float(begin_str))-2209017600)
        return str(time.strftime("%Y-%m-%d", timeArray))

    def get_stop_time(begin_str, stop_str):
        if int(float(stop_str)) != 256000000000:
            timeArray = time.localtime(int(float(stop_str))-2209017600)
            return time.strftime("%Y-%m-%d", timeArray)
        else:
            timeArray = time.localtime(int(float(begin_str))-2209017600)
            time_str = time.strftime("%Y-%m-%d", timeArray).split('-')
            time_str[0] = str(int(time_str[0])+50)
            return '-'.join(time_str)
    preprocessed_data['certValidBegin_date'] = pd.to_datetime(preprocessed_data['certValidBegin'].apply(get_begin_time))
    preprocessed_data['certValidStop_date'] = pd.to_datetime(preprocessed_data.apply(lambda row: get_stop_time(row['certValidBegin'], row['certValidStop']), axis=1))
    preprocessed_data = preprocessed_data.drop(['certValidBegin'], axis=1)
    preprocessed_data = preprocessed_data.drop(['certValidStop'], axis=1)

    # 调整数据格式，并且存储到相应位置
    all_columns = preprocessed_data.columns
    append_columns = []
    for x_id in range(0, 79):
        append_columns.append('x_'+str(x_id))
    lefted_columns = list(set(all_columns)-set(append_columns))
    append_feature = (preprocessed_data[append_columns].astype(np.float64)).astype(np.int64)
    lefted_feature = preprocessed_data[lefted_columns]
    float_columns = ['lmt']
    date_columns = ['certValidBegin_date', 'certValidStop_date']
    int_columns = list(set(lefted_columns)-set(float_columns)-set(date_columns)-set(refer_columns))
    float_feature = lefted_feature[float_columns].astype(np.float64)
    date_feature = lefted_feature[date_columns]
    refer_feature = (lefted_feature[refer_columns].astype(np.float64)).astype(np.int64)
    int_feature = (lefted_feature[int_columns].astype(np.float64)).astype(np.int64)
    base_feature = ((refer_feature.join(int_feature)).join(date_feature)).join(float_feature)
    return base_feature, append_feature


# 产生针对每一条数据的feature
def generate_id_feature(base_feature):
    id_depend_feature = base_feature
    id_depend_feature['certValidBegin_date'] = pd.to_datetime(id_depend_feature['certValidBegin_date'])
    id_depend_feature['certValidStop_date'] = pd.to_datetime(id_depend_feature['certValidStop_date'])
    id_depend_feature['date_weedofday'] = id_depend_feature['certValidBegin_date'].dt.dayofweek
    # id_depend_feature['date_weekend'] = np.where(id_depend_feature['date_weedofday'] >= 5, 1, 0) #不具有相关性
    # id_depend_feature['date_day'] = id_depend_feature['certValidBegin_date'].dt.day #不具有相关性
    id_depend_feature['date_month'] = id_depend_feature['certValidBegin_date'].dt.month
    id_depend_feature['date_year_use'] = id_depend_feature['certValidStop_date'].dt.year - id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature['date_year_use'] = id_depend_feature['date_year_use']//10
    id_depend_feature = id_depend_feature.drop(['certValidBegin_date'], axis=1)
    id_depend_feature = id_depend_feature.drop(['certValidStop_date'], axis=1)

    # id_depend_feature.drop(['isNew'], axis=1, inplace=True)
    # id_depend_feature.drop(['edu'], axis=1, inplace=True)
    # id_depend_feature.drop(['5yearBadloan'], axis=1, inplace=True)
    # id_depend_feature['highest_edu_99'] = np.where(id_depend_feature['highestEdu'] == 99, 1, 0)#无效特征

    id_depend_feature = id_depend_feature.drop(['certId'], axis=1)
    id_depend_feature = id_depend_feature.drop(['dist'], axis=1)
    id_depend_feature = id_depend_feature.drop(['bankCard'], axis=1)
    id_depend_feature = id_depend_feature.drop(['residentAddr'], axis=1)
    return id_depend_feature


def simplify_append_feature(append_feature):
    def num2col(num):
        return 'x_'+str(num)
    need_del_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 22, 23, 24, 37, 38, 39, 40, 58, 59, 60, 78]
    need_del_colname = list(map(num2col, need_del_list))
    for column in need_del_colname:
        append_feature.drop(column, axis=1, inplace=True)
    return append_feature


def add_random_feature(index_feature):
    row_size = index_feature.shape[0]
    for index_range in range(200):
        exp_num = rd.randint(0, 2)
        mult_num = rd.randint(1, 5)
        bias_num = rd.randint(-5, 5)
        simbol = rd.randint(0, 1)
        random_data = np.random.random(row_size)-0.5
        random_data = (random_data*pow(10, exp_num)*mult_num+bias_num)*pow(-1, simbol)
        index_feature['add_%s' % index_range] = random_data.copy()  # 为什么？
    index_feature = index_feature.drop(index_feature.columns[0], axis=1)
    return index_feature


# 存储最终数据
def store_feed(merged_data, feed_path):
    merged_data.to_csv(feed_path, index=False)


def only_read_base_append(feature_dir):
    base_feature = pd.read_csv(feature_dir+'base_feature.csv')
    append_feature = pd.read_csv(feature_dir+'append_feature.csv')
    return base_feature, append_feature


def im_show(feature_map):  # 定义热力图函数
    columns = feature_map.columns
    size = len(columns)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = pearsonr(feature_map[columns[i]], feature_map[columns[j]])[0]
    matrix = np.around(matrix, 3)
    print(columns)
    print(matrix[-1])
    _, ax = plt.subplots(figsize=(50, 38))
    sns.heatmap(matrix, annot=True, vmax=1, vmin=0, square=True)
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.setp(ax.get_yticklabels(), rotation=360)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.show()


def main():
    dataset_names = config.DATASET_NAMES
    # 按照数据集生成feature
    for dataset_name in dataset_names:
        director = config.split_dir+dataset_name+'/'
        feature_dir = director+'feature/'

        base_feature, append_feature = preprocess(pd.read_csv(director+'/data.csv'))
        # base_feature, append_feature = only_read_base_append(feature_dir)

        id_depend_feature = generate_id_feature(base_feature)
        append_feature = simplify_append_feature(append_feature)
        
        merged_feature = id_depend_feature
        merged_feature = merged_feature.join(append_feature)
        '''
        add_feature = add_random_feature(append_feature.iloc[:, [0]])
        merged_feature = merged_feature.join(add_feature)
        '''
        # merged_feature = add_feature  # 假如只有随机数据

        base_feature.to_csv(feature_dir+'base_feature.csv', index=False)
        append_feature.to_csv(feature_dir+'append_feature.csv', index=False)
        id_depend_feature.to_csv(feature_dir+'id_depend_feature.csv', index=False)
        merged_feature.to_csv(feature_dir+'merged_feature.csv', index=False)

        # 存储feed数据
        store_feed(merged_feature, director+'/feed.csv')


def balance(pos_num=None):  # 给定正样本扩充后的数值
    train_dir = 'Datasets/Xiamen_data/split/train/'
    target_dir = 'Datasets/Xiamen_data/split/train/'
    train_feed = pd.read_csv(train_dir+'feed.csv')
    train_target = pd.read_csv(target_dir+'target.csv')
    if pos_num:
        smo = SMOTE(ratio={1: pos_num}, random_state=42)
        train_feed, train_target = smo.fit_sample(train_feed.values, train_target.values)
    else:
        train_feed, train_target = train_feed.values, train_target.values
    columns = pd.read_csv(train_dir+'feed.csv').columns
    train_feed = pd.DataFrame(train_feed, columns=columns)
    train_target = pd.DataFrame(train_target, columns=['target'])
    train_feed.to_csv(train_dir+'feed_b.csv', index=False)
    train_target.to_csv(target_dir+'target_b.csv', index=False)


if __name__ == '__main__':
    main()

    feed = pd.read_csv('Datasets/Xiamen_data/split/train/feed.csv')
    target = pd.read_csv('Datasets/Xiamen_data/split/train/target.csv')
    # im_show(feed.join(target))

    balance()
