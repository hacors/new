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
    toy_df = df.sample(n=10000)
    toy_df.to_csv(toy_dir, index=False)


def data_clean(input_df):
    output_df = pd.DataFrame()

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
    output_df['u_id'] = input_df['User_id'].astype(np.int32)
    output_df['m_id'] = input_df['Merchant_id'].astype(np.int16)
    output_df['c_id'] = (input_df['Coupon_id'].fillna(0)).astype(np.int16)
    output_df['distance'] = (input_df['Distance'].fillna(input_df['Distance'].mean())).astype(np.int8)

    output_df['coupons'] = np.where(input_df['Date_received'].notnull(), True, False)
    output_df['date_coupons'] = (input_df['Date_received'].fillna(input_df['Date']))
    output_df['date_coupons'] = output_df['date_coupons'].apply(float_to_data)
    output_df['consume'] = np.where(input_df['Date'].notnull(), True, False)
    output_df['date_consume'] = (input_df['Date'].fillna(input_df['Date_received']))
    output_df['date_consume'] = output_df['date_consume'].apply(float_to_data)

    output_df['temp'] = input_df['Discount_rate'].fillna('1.0')  # 注意需要填补为字符串
    output_df[['full', 'minus', 'discount_rate']] = (output_df['temp'].apply(get_discount_rate)).apply(pd.Series)
    output_df[['full', 'minus']] = output_df[['full', 'minus']].astype(np.int16)
    output_df['discount_rate'] = output_df['discount_rate'].astype(np.float16)

    output_df.drop(['temp'], axis=1, inplace=True)
    return output_df


if __name__ == '__main__':
    # 分别存储线下和线上的玩具数据集
    # save_toydata(origin_online_dir, origin_online_toy_dir)
    # save_toydata(origin_offline_dir, origin_offline_toy_dir)
    save_toydata(clean_offline_dir, clean_offline_toy_dir)
    pass
