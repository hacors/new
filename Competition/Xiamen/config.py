# 配置全局变量

import os
import shutil


DIR = 'Datasets/Xiamen_data'
DATASET_NAMES = ['train', 'validate', 'test', 'predict']


# 原始数据地址
origin_dir = DIR+'/origin'

origin_train_data_path = origin_dir+'/train.csv'
origin_train_target_path = origin_dir+'/train_target.csv'
origin_test_data_path = origin_dir+'/test.csv'

# 划分了训练集、验证集、测试集的地址，可以按照某种地区提取特征，置于feature文件夹中
# 同时需要给出预测集的处理地址
split_dir = DIR+'/split'

train_dir = split_dir+'/train'
train_feature_dir = split_dir+'/train/feature'
train_data = train_dir+'/data.csv'
train_feed = train_dir+'/feed.cvs'
train_target = train_dir+'/target.csv'

validate_dir = split_dir+'/validate'
validate_feature_dir = split_dir+'/validate/feature'
validate_data = validate_dir+'/data.csv'
validate_feed = validate_dir+'/feed.cvs'
validate_target = validate_dir+'/target.csv'

test_dir = split_dir+'/test'
test_feature_dir = split_dir+'/test/feature'
test_data = test_dir+'/data.csv'
test_feed = test_dir+'/feed.cvs'
test_target = test_dir+'/target.csv'

predict_dir = split_dir+'/predict'
predict_feature_dir = split_dir+'/predict/feature'
predict_data = predict_dir+'/data.csv'
predict_feed = predict_dir+'/feed.csv'
# 注意predict为线上结果，无target

# 模型存储地址
model_dir = DIR+'/model'

# 提交文件
submit_dir = DIR+'/submit'
submit_feature_dir = submit_dir+'/feature'
submit_data = submit_dir+'/data.csv'
submit_feed = submit_dir+'/feed.cvs'
submit_result = submit_dir+'/result.csv'

# 其他数据
SPLIT_VAL_TEST = 30000
SPLIT_TEST_TRAIN = 47833


# 定义生成文件夹函数
def mkdirector(director):
    if os.path.exists(director):
        shutil.rmtree(director)
    else:
        os.mkdir(director)


if __name__ == '__main__':
    pass
