import glob  # 获取文件名列表
import os
import h5py
import numpy as np
import PIL
import scipy
from scipy import io as scio
from scipy import ndimage as scnd
from matplotlib import pyplot as plt


def gaussian_process(p_gt_list: np.array, p_shape):  # 尝试使用发散卷积
    pos_list = p_gt_list.copy()
    corner_pos = np.array([[0, 0], [0, p_shape[1]], [p_shape[0], 0], p_shape])  # 需要扩展，防止人数过少
    dens_array = np.zeros(p_shape)
    if len(pos_list):
        pos_extend = np.concatenate((pos_list, corner_pos), axis=0)
        kd_tree = scipy.spatial.KDTree(pos_extend, leafsize=2048)
        kd_dis, kd_locat = kd_tree.query(pos_list, k=4)
        for index, pos in enumerate(pos_list):
            if pos[0] < p_shape[0] and pos[1] < p_shape[1]:  # 防止越界
                temp_filter = np.zeros(p_shape)
                temp_filter[pos[0], pos[1]] = 1.0
                sigma = (kd_dis[index][1]+kd_dis[index][2]+kd_dis[index][3])*0.1
                dens_array += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    return dens_array


def swap_axis(gt_list):  # 对gt_list翻转并取整，因为图片的坐标与array的坐标不一致
    gt_list = gt_list.tolist()
    result = []
    for pos in gt_list:
        result.append(pos[::-1])
    result = np.array(result).astype(np.int)
    return result


def get_dataset_mall(root_path):
    '''
    返回mall数据集的所有数据，分为训练数据集和测试数据集，各自按照视频片段分割
    每个视频片段包括图片列表和gt列表
    按照统一的命名规则将所有数据处理，存放在指定的文件夹procesed_data中
    训练集/测试集_视频片段编号_图片编号.h5
    例如video0_pic0.h5
    文件中保存原图片，gt列表，密度图
    '''
    frame_path = os.path.join(root_path, 'frames')
    processed_path = os.path.join(root_path, 'train_processed')
    pic_list = glob.glob(os.path.join(frame_path, '*.jpg'))
    gt_path = os.path.join(root_path, 'mall_gt.mat')
    gt_file = scio.loadmat(gt_path)  # gt_list = gt_file['frame'][0][pic_index][0, 0][0]
    video_index = 0
    for pic_index in range(len(pic_list)):
        single_pic = PIL.Image.open(pic_list[pic_index])
        single_pic = np.array(single_pic)
        single_gt = gt_file['frame'][0][pic_index][0, 0][0]
        single_gt = swap_axis(single_gt)
        single_dens = gaussian_process(single_gt, single_pic.shape[:2])
        file_name = 'video_%s_pic_%s_.h5' % (str(video_index).zfill(5), str(pic_index).zfill(5))
        h5_path = os.path.join(processed_path, file_name)
        with h5py.File(h5_path, 'w') as h5_file:
            h5_file['pic'] = single_pic
            h5_file['gt'] = single_gt
            h5_file['dens'] = single_dens


def get_dataset_expo2010(root_path):
    '''
    返回expo2010数据集的所有数据，分为训练数据集和测试数据集，各自按照视频片段分割
    每个视频片段包括图片列表和gt列表
    按照统一的命名规则将所有数据处理，存放在指定的文件夹procesed_data中
    训练集/测试集_视频片段编号_图片编号.h5
    例如video0_pic0.h5
    文件中保存原图片，gt列表，密度图
    '''
    for set_type in ['train', 'test']:
        processed_path = os.path.join(root_path, '%s_processed' % set_type)
        frame_path = os.path.join(root_path, '%s_frame' % set_type)
        label_path = os.path.join(root_path, '%s_label' % set_type)
        label_type = glob.glob(os.path.join(label_path, '*'))
        for video_index in range(88, len(label_type)):
            video_path = label_type[video_index]
            mat_list = glob.glob(os.path.join(video_path, '*.mat'))
            for mat_index in range(0, len(mat_list)-1):
                pic_index = mat_index
                mat_path = mat_list[mat_index]
                pic_name = mat_path.split('\\')[-1].replace('.mat', '.jpg')
                pic_path = os.path.join(frame_path, pic_name)
                single_pic = PIL.Image.open(pic_path)
                single_pic = np.array(single_pic)
                try:
                    gt_file = scio.loadmat(mat_path)
                    single_gt = gt_file['point_position']
                except Exception:
                    featrue = h5py.File(mat_path, 'r')
                    gt_file = featrue['point_position']
                    if len(gt_file.shape) == 1:
                        single_gt = np.empty((1, 2))
                    else:
                        flatten_gt = np.ravel(gt_file.value, order='F')
                        single_gt = np.reshape(flatten_gt, (gt_file.shape[::-1]))
                single_gt = swap_axis(single_gt)
                single_dens = gaussian_process(single_gt, single_pic.shape[:2])
                file_name = 'video_%s_pic_%s_.h5' % (str(video_index).zfill(5), str(pic_index).zfill(5))
                h5_path = os.path.join(processed_path, file_name)
                with h5py.File(h5_path, 'w') as h5_file:
                    h5_file['pic'] = single_pic
                    h5_file['gt'] = single_gt
                    h5_file['dens'] = single_dens


def check_file(file_path):
    h5_file = h5py.File(file_path, 'r')
    pic = h5_file['pic']
    dens = h5_file['dens']
    plt.imshow(dens)
    plt.show()
    plt.imshow(pic)
    plt.show()


if __name__ == '__main__':
    get_dataset_mall(os.path.join('Datasets', 'mall'))
    get_dataset_expo2010(os.path.join('Datasets', 'expo2010'))
