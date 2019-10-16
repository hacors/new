import config
import pandas as pd
import time
import numpy as np
from imblearn.over_sampling import SMOTE


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
    id_depend_feature['date_weekend'] = np.where(id_depend_feature['date_weedofday'] >= 5, 1, 0)
    id_depend_feature['date_day'] = id_depend_feature['certValidBegin_date'].dt.day
    id_depend_feature['date_month'] = id_depend_feature['certValidBegin_date'].dt.month
    # id_depend_feature['date_year'] = id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature['date_year_use'] = id_depend_feature['certValidStop_date'].dt.year - id_depend_feature['certValidBegin_date'].dt.year
    id_depend_feature['date_year_use'] = id_depend_feature['date_year_use']//10
    id_depend_feature = id_depend_feature.drop(['certValidBegin_date'], axis=1)
    id_depend_feature = id_depend_feature.drop(['certValidStop_date'], axis=1)

    # id_depend_feature.drop(['isNew'], axis=1, inplace=True)
    # id_depend_feature.drop(['edu'], axis=1, inplace=True)
    id_depend_feature.drop(['5yearBadloan'], axis=1, inplace=True)
    id_depend_feature['highest_edu_99'] = np.where(id_depend_feature['highestEdu'] == 99, 1, 0)

    id_depend_feature = id_depend_feature.drop(['certId'], axis=1)
    id_depend_feature = id_depend_feature.drop(['dist'], axis=1)
    id_depend_feature = id_depend_feature.drop(['bankCard'], axis=1)
    id_depend_feature = id_depend_feature.drop(['residentAddr'], axis=1)
    return id_depend_feature


# 存储最终数据
def store_feed(merged_data, feed_path):
    merged_data.to_csv(feed_path, index=False)


def only_read(feature_dir):
    base_feature = pd.read_csv(feature_dir+'base_feature.csv')
    append_feature = pd.read_csv(feature_dir+'append_feature.csv')
    return base_feature, append_feature


def main():
    dataset_names = config.DATASET_NAMES
    # 按照数据集生成feature
    for dataset_name in dataset_names:
        director = config.split_dir+dataset_name+'/'
        feature_dir = director+'feature/'

        base_feature, append_feature = preprocess(pd.read_csv(director+'/data.csv'))
        # base_feature, append_feature = only_read(feature_dir)

        id_depend_feature = generate_id_feature(base_feature)
        # merged_feature = id_depend_feature.join(append_feature)
        merged_feature = id_depend_feature
        base_feature.to_csv(feature_dir+'base_feature.csv', index=False)
        append_feature.to_csv(feature_dir+'append_feature.csv', index=False)
        id_depend_feature.to_csv(feature_dir+'id_depend_feature.csv', index=False)
        merged_feature.to_csv(feature_dir+'merged_feature.csv', index=False)

        # 存储feed数据
        store_feed(merged_feature, director+'/feed.csv')


def balance():
    train_dir = 'Datasets/Xiamen_data/split/train/'
    target_dir = 'Datasets/Xiamen_data/split/train/'
    train_feed = pd.read_csv(train_dir+'feed.csv')
    train_target = pd.read_csv(target_dir+'target.csv')
    '''
    smo = SMOTE(ratio={1: 10000}, random_state=42)
    train_feed, train_target = smo.fit_sample(pd.read_csv(train_feed, train_target))
    columns = pd.read_csv(train_dir+'feed.csv').columns
    train_feed = pd.DataFrame(train_feed, columns=columns)
    train_target = pd.DataFrame(train_target, columns=['target'])
    '''
    train_feed.to_csv(train_dir+'feed_b.csv', index=False)
    train_target.to_csv(target_dir+'target_b.csv', index=False)


if __name__ == '__main__':
    main()
    balance()
