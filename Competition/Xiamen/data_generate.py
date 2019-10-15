import config
import pandas as pd
import time
import numpy as np


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
    base_feature = (int_feature.join(date_feature)).join(float_feature)
    return base_feature, append_feature, refer_feature


# 产生针对每一条数据的feature
def generate_id_feature(base_feature):
    id_depend_feature = base_feature
    id_depend_feature['certValidBegin_date'] = pd.to_datetime(id_depend_feature['certValidBegin_date'])
    id_depend_feature['certValidStop_date'] = pd.to_datetime(id_depend_feature['certValidStop_date'])
    id_depend_feature['date_weedofday'] = id_depend_feature['certValidBegin_date'].dt.dayofweek
    id_depend_feature['date_weekend'] = np.where(id_depend_feature['date_weedofday'] >= 5, 1, 0)
    id_depend_feature['date_day'] = id_depend_feature['certValidBegin_date'].dt.day
    id_depend_feature['date_month'] = id_depend_feature['certValidBegin_date'].dt.month
    # id_depend_feature['date_year'] = id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature['date_year_use'] = id_depend_feature['certValidStop_date'].dt.year - id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature = id_depend_feature.drop(['certValidBegin_date'], axis=1)
    id_depend_feature = id_depend_feature.drop(['certValidStop_date'], axis=1)
    return id_depend_feature


def generate_group_feature(group_id_feature, group_target, group_refer_feature):
    depend_feature = group_id_feature.join(group_target)
    group_feature_list = []
    for refer_name in refer_columns:
        group_feature_input = group_refer_feature[[refer_name]]
        group_feature_input = group_feature_input.join(depend_feature['target'])
        # 依据depend添加其他feature
        grouped_data = group_feature_input.groupby([refer_name])
        group_feature_mean = grouped_data.mean()
        '''
        group_feature_sum = grouped_data.sum()
        if refer_name in ['bankCard', 'residentAddr']:  # 存在大量空值的列对应的sum属性去除，即对group之后的target属性不需要计算sum值
            group_feature_sum = group_feature_sum.drop(['target'], axis=1)
        suf = ('_%s_sum' % refer_name, '_%s_mean' % refer_name)
        group_feature_output = pd.merge(group_feature_sum, group_feature_mean, left_index=True, right_index=True, suffixes=suf)
        '''
        group_feature_output = group_feature_mean
        group_feature_list.append(group_feature_output)
    return group_feature_list


# 由预处理的数据和特征数据融合成最终的数据，并且存储最终数据
def merge(id_depend_feature, refer_feature, group_feature_list):
    merged_feature = refer_feature
    for index in range(len(refer_columns)):
        refer_name = refer_columns[index]
        group_feature = group_feature_list[index]
        refer_feature = pd.merge(refer_feature, group_feature, left_on=refer_name, right_index=True, how='left')
        refer_feature = refer_feature.drop([refer_name], axis=1)
    merged_feature = merged_feature.join(refer_feature)

    # 以均值填补所有的空值
    columns_havena = list(merged_feature.isnull().any())
    columns_names = merged_feature.columns
    for index, havena in enumerate(columns_havena):
        if havena:
            merged_feature[columns_names[index]].fillna(merged_feature[columns_names[index]].mean(), inplace=True)
    return merged_feature


# 存储最终数据
def store_feed(merged_data, feed_path):
    merged_data.to_csv(feed_path, index=False)


def only_read(feature_dir):
    base_feature = pd.read_csv(feature_dir+'base_feature.csv')
    append_feature = pd.read_csv(feature_dir+'append_feature.csv')
    refer_feature = pd.read_csv(feature_dir+'refer_feature.csv')
    return base_feature, append_feature, refer_feature


def main():
    dataset_names = config.DATASET_NAMES
    # 首先生成临时数据
    group_base_feature, _, group_refer_feature = preprocess(pd.read_csv(config.train_data))
    group_id_feature = generate_id_feature(group_base_feature)
    group_target = pd.read_csv(config.train_target)
    group_feature_list = generate_group_feature(group_id_feature, group_target, group_refer_feature)
    # 存储featurelist
    config.mkdirector(config.global_feature_dir)
    for index in range(len(refer_columns)):
        refer_name = refer_columns[index]
        group_feature = group_feature_list[index]
        group_feature.to_csv(config.global_feature_dir+refer_name+'.csv', index=False)

    # 按照数据集生成feature
    for dataset_name in dataset_names:
        director = config.split_dir+dataset_name+'/'
        feature_dir = director+'feature/'

        data_path = director+'/data.csv'
        feed_path = director+'/feed.csv'

        config.mkdirector(feature_dir)
        base_feature, append_feature, refer_feature = preprocess(pd.read_csv(data_path))

        # base_feature, append_feature, refer_feature = only_read(feature_dir)
        base_feature.to_csv(feature_dir+'base_feature.csv', index=False)
        append_feature.to_csv(feature_dir+'append_feature.csv', index=False)
        refer_feature.to_csv(feature_dir+'refer_feature.csv', index=False)
        id_depend_feature = generate_id_feature(base_feature)
        id_depend_feature.to_csv(feature_dir+'id_depend_feature.csv', index=False)
        # merged_feature = merge(id_depend_feature, append_feature, refer_feature, group_feature_list)
        merged_feature = id_depend_feature

        merged_feature = merged_feature.join(append_feature)
        merged_feature.to_csv(feature_dir+'merged_feature.csv', index=False)
        # 存储feed数据
        store_feed(merged_feature, feed_path)


if __name__ == '__main__':
    main()
