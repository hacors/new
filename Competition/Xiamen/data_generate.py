import config
import pandas as pd

# 数据预处理并返回处理好的数据


def preprocess(data):
    preprocessed_data = data
    # 基础处理,去除无关数据
    preprocessed_data = preprocessed_data.drop(['id', 'isNew'], axis=1)
    return preprocessed_data


# 获取数据中的各种特征并且存储特征数据表
def generate_feature(preprocessed_data, feature_dir):
    pass


# 由预处理的数据和特征数据融合成最终的数据，并且存储最终数据
def merge(preprocessed_data, feature_dir):
    merged_data = preprocessed_data
    # 后续添加feature
    return merged_data


# 存储最终数据
def store_feed(merged_data, feed_path):
    merged_data.to_csv(feed_path, index=False)


def main():
    dataset_names = config.DATASET_NAMES
    for dataset_name in dataset_names:
        director = config.split_dir+'/'+dataset_name
        feature_dir = director+'/feature'
        data_path = director+'/data.csv'
        feed_path = director+'/feed.csv'

        origin_data = pd.read_csv(data_path)
        preprocessed_data = preprocess(origin_data)
        generate_feature(preprocessed_data, feature_dir)
        merged_data = merge(preprocessed_data, feature_dir)
        store_feed(merged_data, feed_path)


if __name__ == '__main__':
    main()
