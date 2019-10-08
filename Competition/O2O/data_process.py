import pandas as pd
import numpy as np
DATASET_ROOT = 'Datasets/O2O'

# 原始数据
origin_online_dir = DATASET_ROOT+'/Origin/ccf_online_stage1_train.csv'
origin_offline_dir = DATASET_ROOT+'/Origin/ccf_offline_stage1_train.csv'
origin_online_toy_dir = DATASET_ROOT+'/Origin/online_toy.csv'
origin_offline_toy_dir = DATASET_ROOT+'/Origin/offline_toy.csv'
origin_test_dir = DATASET_ROOT+'/Origin/ccf_offline_stage1_test_revised.csv'
# 清理后的数据
clean_offline_dir = DATASET_ROOT+'/Clean/offline.csv'
clean_offline_toy_dir = DATASET_ROOT+'/Clean/offline_toy.csv'


def save_toydata(origin_dir, toy_dir):  # 存储玩具数据
    df = pd.read_csv(origin_dir)
    toy_df = df[:10000]
    toy_df.to_csv(toy_dir, index=False)


def data_clean(df_input):  # 初步处理，填补空值，将数据转换为可计算形式
    df_output = pd.DataFrame()

    def float_to_data(row):
        row = str(int(row))
        row = row[:4]+'-'+row[4:6]+'-'+row[6:]
        row = pd.to_datetime(row)
        return row

    def get_discount_rate(row):
        if ':' in row:
            full, minus = list(map(int, row.split(':')))
            return full, minus, (full-minus)/full
        else:
            return 0, 0, row
    df_output['u_id'] = df_input['User_id'].astype(np.int32)
    df_output['m_id'] = df_input['Merchant_id'].astype(np.int16)
    df_output['c_id'] = (df_input['Coupon_id'].fillna(0)).astype(np.int16)
    df_output['distance'] = (df_input['Distance'].fillna(df_input['Distance'].mean())).astype(np.int8)

    df_output['coupons'] = np.where(df_input['Date_received'].notnull(), True, False)
    df_output['date_coupons'] = (df_input['Date_received'].fillna(df_input['Date']))
    df_output['date_coupons'] = df_output['date_coupons'].apply(float_to_data)
    df_output['consume'] = np.where(df_input['Date'].notnull(), True, False)
    df_output['date_consume'] = (df_input['Date'].fillna(df_input['Date_received']))
    df_output['date_consume'] = df_output['date_consume'].apply(float_to_data)

    df_output['temp'] = df_input['Discount_rate'].fillna('1.0')  # 注意需要填补为字符串
    df_output[['full', 'minus', 'discount_rate']] = (df_output['temp'].apply(get_discount_rate)).apply(pd.Series)
    df_output[['full', 'minus']] = df_output[['full', 'minus']].astype(np.int16)
    df_output['discount_rate'] = df_output['discount_rate'].astype(np.float16)

    df_output.drop(['temp'], axis=1, inplace=True)
    return df_output


def add_unuse_consume_date(df_input):
    # 添加用户对商店 领取优惠券之后紧接着的消费日期
    df_sorted = df_input.sort_values('u_id')
    # print(df_sorted.head(30))
    df_grouped = df_sorted.groupby('u_id').count()
    index_list = list(df_grouped['m_id'])
    global df_cutted  # 需要声明为全局变量，改变全局的取值
    df_output = pd.DataFrame()

    def get_consumedate_after_coupons(m_id, coupons, consume, date):
        if coupons and not consume:  # 只有获取了优惠券并且没有使用的部分才需要计算
            df_temp = df_cutted['date_consume'][(df_cutted['m_id'] == m_id) & (df_cutted['coupons'] == False) & (df_cutted['date_consume'] > date)]
            return df_temp.min()
    start_index = 0
    for add in index_list:
        end_index = start_index+add
        df_cutted = df_sorted[start_index:end_index]
        df_cutted_temp = df_cutted.copy()  # 需要一个cutted的副本接收输出结果
        df_cutted_temp['unuse_consume_date'] = df_cutted.apply(lambda row: get_consumedate_after_coupons(row['m_id'], row['coupons'], row['consume'], row['date_coupons']), axis=1)
        df_output = df_output.append(df_cutted_temp)
        start_index = end_index
    df_output.sort_index(axis=0)
    return df_output


if __name__ == '__main__':
    # 分别存储线下和线上的玩具数据集
    df_origin_offline = pd.read_csv(origin_offline_dir)
    save_toydata(origin_offline_dir, origin_offline_toy_dir)
    df_origin_offline_toy = pd.read_csv(origin_offline_toy_dir)

    '''
    df_clean_offline = data_clean(df_origin_offline)
    df_clean_offline.to_csv(clean_offline_dir, index=False)
    '''
    df_clean_offline = pd.read_csv(clean_offline_dir)
    save_toydata(clean_offline_dir, clean_offline_toy_dir)
    df_clean_offline_toy = pd.read_csv(clean_offline_toy_dir)
