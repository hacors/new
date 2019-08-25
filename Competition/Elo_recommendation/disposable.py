import pandas as pd
# 读取数据
DATA_ROOT = 'Datasets/Elo/'
df_train = pd.read_csv(DATA_ROOT+'train.csv')
df_test = pd.read_csv(DATA_ROOT+'test.csv')
df_history_trac = pd.read_csv(DATA_ROOT+'historical_transactions.csv')
df_new_trac = pd.read_csv(DATA_ROOT+'new_merchant_transactions.csv')
df_merchant = pd.read_csv(DATA_ROOT+'merchants.csv')
# 提取模板数据
df_history_trac_cut = df_history_trac[0:10000]
df_new_trac_cut = df_new_trac[0:10000]
df_merchant_cut = df_merchant[0:10000]
df_history_trac_cut.to_csv(DATA_ROOT+'historical_transactions_cut.csv')
df_new_trac_cut.to_csv(DATA_ROOT+'new_merchant_transactions_cut.csv')
df_merchant_cut.to_csv(DATA_ROOT+'merchants_cut.csv')
