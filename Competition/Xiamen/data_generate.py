import config
import pandas as pd
import time
import numpy as np


refer_columns = ['certId', 'dist', 'bankCard', 'residentAddr']


# 数据预处理并返回处理好的数据
def preprocess(data, feature_dir):
    preprocessed_data = data
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

    temp = pd.merge(int_feature, date_feature, left_index=True, right_index=True)
    origin_feature = pd.merge(temp, float_feature, left_index=True, right_index=True)

    append_feature.to_csv(feature_dir+'append_feature.csv', index=False)  # 所有的x数据
    refer_feature.to_csv(feature_dir+'refer_feature.csv', index=False)  # 分组feature的分组参照
    origin_feature.to_csv(feature_dir+'origin_feature.csv', index=False)  # 原始feature
    return origin_feature


# 产生针对每一条数据的feature
def generate_id_feature(origin_feature, feature_dir):
    id_depend_feature = origin_feature

    id_depend_feature['certValidBegin_date'] = pd.to_datetime(id_depend_feature['certValidBegin_date'])
    id_depend_feature['certValidStop_date'] = pd.to_datetime(id_depend_feature['certValidStop_date'])
    id_depend_feature['date_weedofday'] = id_depend_feature['certValidBegin_date'].dt.dayofweek
    id_depend_feature['date_weekend'] = np.where(id_depend_feature['date_weedofday'] >= 5, 1, 0)
    id_depend_feature['date_day'] = id_depend_feature['certValidBegin_date'].dt.day
    id_depend_feature['date_month'] = id_depend_feature['certValidBegin_date'].dt.month
    id_depend_feature['date_year_use'] = id_depend_feature['certValidStop_date'].dt.year - id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature = id_depend_feature.drop(['certValidBegin_date'], axis=1)
    id_depend_feature = id_depend_feature.drop(['certValidStop_date'], axis=1)

    id_depend_feature.to_csv(feature_dir+'id_depend_feature.csv', index=False)


# 获取数据中的各种特征并且存储特征数据表
def generate_certId_feature(id_depend_feature, feature_dir, group_feature, labels):
    pass


def generate_dist_feature(id_depend_feature, feature_dir, group_feature, labels):
    pass


def generate_bankCard_feature(id_depend_feature, feature_dir, group_feature, labels):
    pass


def generate_residentAddr_feature(id_depend_feature, feature_dir, group_feature, labels):
    pass


# 由预处理的数据和特征数据融合成最终的数据，并且存储最终数据
def merge(id_depend_feature, refer_feature, feature_list, append_feature):
    merged_feature = id_depend_feature

    merged_feature = pd.merge(merged_feature, append_feature, left_index=True, right_index=True)
    merged_feature = pd.merge(merged_feature, refer_feature, left_index=True, right_index=True)
    return merged_feature


# 存储最终数据
def store_feed(merged_data, feed_path):
    merged_data.to_csv(feed_path, index=False)


def main():
    dataset_names = config.DATASET_NAMES
    # 首先生成普通feature
    for dataset_name in dataset_names:
        director = config.split_dir+dataset_name+'/'
        feature_dir = director+'feature/'
        config.mkdirector(feature_dir)
        data_path = director+'/data.csv'
        feed_path = director+'/feed.csv'
        origin_data = pd.read_csv(data_path)
        origin_feature = preprocess(origin_data, feature_dir)
        generate_id_feature(origin_feature, feature_dir)

    # 按照训练集和真实标签生成global_feature
    global_feature_dir = config.global_feature_dir
    config.mkdirector(global_feature_dir)
    global_base_feature = pd.merge(pd.read_csv(config.train_dir+'feature/id_depend_feature.csv'), pd.read_csv(config.train_dir+'target.csv'), left_index=True, right_index == True)
    global_refer_feature = pd.read_csv(config.train_dir+'feature/refer_feature.csv')

    generate_certId_feature(global_base_feature, global_feature_dir, global_refer_feature['certId'])
    generate_dist_feature(global_base_feature, global_feature_dir, global_refer_feature['dist'])
    generate_bankCard_feature(global_base_feature, global_feature_dir, global_refer_feature['bankCard'])
    generate_residentAddr_feature(global_base_feature, global_feature_dir, global_refer_feature['residentAddr'])

    # 融合数据
    '''
    for dataset_name in dataset_names:
        director = config.split_dir+dataset_name+'/'
        feature_dir = director+'feature/'
        config.mkdirector(feature_dir)
        data_path = director+'/data.csv'
        feed_path = director+'/feed.csv'
        origin_data = pd.read_csv(data_path)
        origin_feature = preprocess(origin_data, feature_dir)
        generate_id_feature(origin_feature, feature_dir)
        merged_data = merge(id_depend_feature, refer_feature, [certId_depend_feature, dist_depend_feature, bankCard_depend_feature, residentAddr_depend_feature], append_feature)
    '''
    store_feed(merged_data, feed_path)


if __name__ == '__main__':
    main()
