from scipy import io as scio
import numpy as np
import glob  # 获取文件名列表


def gaussian_process(p_gt_list, p_shape):
    '''
    由gt生成密度图
    '''
    pass
def process

def get_dataset_mall(root_path):
    '''
    返回mall数据集的所有数据，分为训练数据集和测试数据集，各自按照视频片段分割
    每个视频片段包括图片列表和gt列表
    按照统一的命名规则将所有数据处理，存放在指定的文件夹中
    训练集/测试集_视频片段编号_图片编号.h5
    文件中只保存图片信息和gt列表
    '''
    fram_path = root_path+'/frames'
    gt_path = root_path+'/mall_gt.mat'
    gt_file = scio.loadmat(gt_path)  # gt_list = gt_file['frame'][0][index][0, 0][0]
    pass


if __name__ == '__main__':
    mall_path = 'Datasets/mall'
    get_dataset_mall(mall_path)
