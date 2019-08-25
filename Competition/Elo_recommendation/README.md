网站：https://www.kaggle.com/c/elo-merchant-category-recommendation/overview

数据结构举例：
Example of historical_transactions.csv:
authorized_flag,card_id,city_id,category_1,installments,category_3,merchant_category_id,merchant_id,month_lag,purchase_amount,purchase_date,category_2,state_id,subsector_id
Y,C_ID_4e6213e9bc,88,N,0,A,80,M_ID_e020e9b302,-8,-0.70333091,2017-06-25 15:33:07,1.00000000,16,37
Y,C_ID_4e6213e9bc,88,N,0,A,367,M_ID_86ec983688,-7,-0.73312848,2017-07-15 12:10:45,1.00000000,16,16
Y,C_ID_4e6213e9bc,88,N,0,A,80,M_ID_979ed661fc,-6,-0.72038600,2017-08-09 22:04:29,1.00000000,16,37

Example of merchants.csv:
merchant_id,merchant_group_id,merchant_category_id,subsector_id,numerical_1,numerical_2,category_1,most_recent_sales_range,most_recent_purchases_range,avg_sales_lag3,avg_purchases_lag3,active_months_lag3,avg_sales_lag6,avg_purchases_lag6,active_months_lag6,avg_sales_lag12,avg_purchases_lag12,active_months_lag12,category_4,city_id,state_id,category_2   
M_ID_838061e48c,8353,792,9,-0.05747065,-0.05747065,N,E,E,-0.40000000,9.66666667,3,-2.25000000,18.66666667,6,-2.32000000,13.91666667,12,N,242,9,1.00000000
M_ID_9339d880ad,3184,840,20,-0.05747065,-0.05747065,N,E,E,-0.72000000,1.75000000,3,-0.74000000,1.29166667,6,-0.57000000,1.68750000,12,N,22,16,1.00000000
M_ID_e726bbae1e,447,690,1,-0.05747065,-0.05747065,N,E,E,-82.13000000,260.00000000,2,-82.13000000,260.00000000,2,-82.13000000,260.00000000,2,N,-1,5,5.00000000

Example of new_merchant_transactions.csv:
authorized_flag,card_id,city_id,category_1,installments,category_3,merchant_category_id,merchant_id,month_lag,purchase_amount,purchase_date,category_2,state_id,subsector_id
Y,C_ID_415bb3a509,107,N,1,B,307,M_ID_b0c793002c,1,-0.55757375,2018-03-11 14:57:36,1.00000000,9,19
Y,C_ID_415bb3a509,140,N,1,B,307,M_ID_88920c89e8,1,-0.56957993,2018-03-19 18:53:37,1.00000000,9,19
Y,C_ID_415bb3a509,330,N,1,B,507,M_ID_ad5237ef6b,2,-0.55103721,2018-04-26 14:08:44,1.00000000,9,14

Example of train.csv:
first_active_month,card_id,feature_1,feature_2,feature_3,target
2017-06,C_ID_92a2005557,5,2,1,-0.82028260
2017-01,C_ID_3d0044924f,4,1,0,0.39291325
2016-08,C_ID_d639edf6cd,2,2,0,0.68805599

Example of test.csv:
first_active_month,card_id,feature_1,feature_2,feature_3
2017-04,C_ID_0ab67a22ab,3,3,1
2017-01,C_ID_130fd0cbdd,2,3,0
2017-08,C_ID_b709037bc5,5,1,1

Example of sample_submission.csv
card_id,target
C_ID_0ab67a22ab,0
C_ID_130fd0cbdd,0
C_ID_b709037bc5,0

参考代码：https://www.kaggle.com/chauhuynh/my-first-kernel-3-699