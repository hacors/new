import glob
import multiprocessing as multp
import os

import numpy as np
import scipy
from scipy import io as scio
from scipy import ndimage as scnd
import tensorflow as tf
import PIL

ROOT = 'Datasets'


def gaussian_filter_density(p_gt_matrix, p_dens_path, p_index):  # 将人群点矩阵图转换为高斯密度图，并保存在指定的文件路径中
    dens_array = np.zeros(p_gt_matrix.shape, dtype=np.float32)
    pos_list = np.array(list(zip(np.nonzero(p_gt_matrix)[0].flatten(), np.nonzero(p_gt_matrix)[1].flatten())))
    kd_tree = scipy.spatial.KDTree(pos_list.copy(), leafsize=2048)
    kd_dis, kd_locat = kd_tree.query(pos_list, k=4)
    if len(pos_list) == 1:
        temp_filter = np.zeros(p_gt_matrix.shape, dtype=np.float32)
        temp_filter[pos_list[0][0], pos_list[0][1]] = 1.
        sigma = np.average(np.array(p_gt_matrix.shape))/4.
        dens_array += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    else:
        for i, pos in enumerate(pos_list):
            temp_filter = np.zeros(p_gt_matrix.shape, dtype=np.float32)
            temp_filter[pos[0], pos[1]] = 1.
            sigma = (kd_dis[i][1]+kd_dis[i][2]+kd_dis[i][3])*0.1  # 这考虑了一种透视折衷
            dens_array += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    scipy.misc.imsave(p_dens_path, dens_array)
    print('finish:', p_index)


def get_shtech_path():  # 获取所有shtech数据集的图片路径
    temp_root = os.path.join(ROOT, 'shtech')
    all_image_path = []
    all_set_path = []
    for p_index, part_label in enumerate(['A', 'B']):
        all_image_path.append([])
        all_set_path.append([])
        for d_index, data_class_label in enumerate(['train', 'test']):
            all_image_path[p_index].append([])
            # 确定好了结构
            os_path = os.path.join(temp_root, 'part_%s_final' % part_label, '%s_data' % data_class_label)
            all_set_path[p_index].append(os_path)
            try:
                os.mkdir(os.path.join(os_path, 'densimg'))
            except Exception:
                pass
            for ima_path in glob.glob(os.path.join(os_path, 'images', '*.jpg')):
                all_image_path[p_index][d_index].append(ima_path)
    return all_image_path, all_set_path


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    # shtech
    shtech_image_path, shtech_set_path = get_shtech_path()
    '''
    for image_part in shtech_image_path:  # 获取所有的密度图
        for image_class in image_part:
            pool = multp.Pool(processes=12)
            for index, image_path in enumerate(image_class):
                gt_path = image_path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
                dens_path = image_path.replace('images', 'densimg').replace('IMG', 'DENS')
                mat = scio.loadmat(gt_path)
                gt_list = mat['image_info'][0, 0][0, 0][0]  # 注意gt的坐标是笛卡尔坐标
                gt_list_int = gt_list.astype(np.int)
                image = PIL.Image.open(image_path)
                gt_matrix = np.zeros(image.shape[:2])
                for gt in gt_list_int:
                    if gt[1] < image.shape[0] and gt[0] < image.shape[1]:
                        gt_matrix[gt[1], gt[0]] = 1.
                # gaussian_filter_density(gt_matrix, dens_path, index)
                pool.apply_async(gaussian_filter_density, (gt_matrix, dens_path, index,))
            pool.close()
            pool.join()
    '''
    for part_index, set_part in enumerate(shtech_set_path):  # 生成tfrecord文件
        for class_index, set_class in enumerate(set_part):
            record_path = os.path.join(set_class, 'all_data.tfrecords')
            image_paths = shtech_image_path[part_index][class_index]
            writer = tf.python_io.TFRecordWriter(record_path)  # tfrecords的写法
            for img_p in image_paths:
                dens_p = img_p.replace('images', 'densimg').replace('IMG', 'DENS')
                img_file = PIL.Image.open(img_p)
                img_array = np.array(img_file)
                img_raw = img_array.tostring()
                dens_file = PIL.Image.open(dens_p)
                dens_array = np.array(dens_file)
                dens_raw = dens_array.tostring()
                shape = img_array.shape
                feature = {
                    'height': int64_feature(shape[0]),
                    'width': int64_feature(shape[1]),
                    'img': bytes_feature(img_raw),
                    'dens': bytes_feature(dens_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            print(set_class)
