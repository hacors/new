import pandas as pd
import datetime
# 读取数据
DATA_ROOT = 'Datasets/Elo/'
df_train = pd.read_csv(DATA_ROOT+'train.csv')
df_test = pd.read_csv(DATA_ROOT+'test.csv')
'''
df_history_trac = pd.read_csv(DATA_ROOT+'historical_transactions.csv')
df_new_trac = pd.read_csv(DATA_ROOT+'new_merchant_transactions.csv')
df_merchant = pd.read_csv(DATA_ROOT+'merchants.csv')
'''
df_history_trac = pd.read_csv(DATA_ROOT+'historical_transactions_cut.csv')
df_new_trac = pd.read_csv(DATA_ROOT+'new_merchant_transactions_cut.csv')
df_merchant = pd.read_csv(DATA_ROOT+'merchants_cut.csv')
# 去除空值
for df in [df_history_trac, df_new_trac, df_merchant]:
    df.dropna(axis=0, how='any', inplace=True)
# 添加数据项
for df in [df_history_trac, df_new_trac]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
# 确定有用的数据项并且确定这些数据的统计特征（因为一张信用卡可以有多条数据）

pass
