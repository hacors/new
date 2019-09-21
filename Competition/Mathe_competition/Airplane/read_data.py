import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
DATASET_ROOT = r'D:\code\Datasets\huawei_airplane'
COLUMN_NAMES = ['id', 'x', 'y', 'z', '', '第三问点标记']


def get_original_data(orig_dir):
    datas = pd.read_csv(orig_dir, header=None, names=COLUMN_NAMES)
    print(datas.head())
    return datas


if __name__ == '__main__':
    data = get_original_data(os.path.join(DATASET_ROOT, 'dataset_2.csv'))
