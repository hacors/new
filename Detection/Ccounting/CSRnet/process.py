import glob
import multiprocessing as multp
import os
import random

import numpy as np
import PIL
import scipy
import tensorflow as tf
from scipy import io as scio
from scipy import ndimage as scnd

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
    np.save(p_dens_path, dens_array)
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
                os.mkdir(os.path.join(os_path, 'dens_np'))
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


def crop_function(img_array: np.array, dens_array: np.array, r_times):  # 将原图裁剪
    shape = img_array.shape
    height, width = int(shape[0]/2), int(shape[1]/2)
    img_crops = [img_array[:height, :width], img_array[height:height*2, :width], img_array[:height, width:width*2], img_array[height:height*2, width:width*2]]
    dens_crops = [dens_array[:height, :width], dens_array[height:height*2, :width], dens_array[:height, width:width*2], dens_array[height:height*2, width:width*2]]
    for r_t in range(r_times):
        d_hei = random.randint(0, height-1)
        d_wid = random.randint(0, width-1)
        img_crops.append(img_array[d_hei:d_hei+height, d_wid:d_wid+width])
        dens_crops.append(dens_array[d_hei:d_hei+height, d_wid:d_wid+width])
    return img_crops, dens_crops, height, width


if __name__ == "__main__":
    # shtech
    shtech_image_path, shtech_set_path = get_shtech_path()

    for image_part in shtech_image_path:  # 获取所有的密度图
        for image_class in image_part:
            pool = multp.Pool(processes=12)
            for index, image_path in enumerate(image_class):
                gt_path = image_path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
                dens_path = image_path.replace('images', 'dens_np').replace('IMG', 'numpy').replace('.jpg', '.npy')
                mat = scio.loadmat(gt_path)
                gt_list = mat['image_info'][0, 0][0, 0][0]  # 注意gt的坐标是笛卡尔坐标
                gt_list_int = gt_list.astype(np.int)
                image = PIL.Image.open(image_path)
                gt_matrix = np.zeros([image.size[1], image.size[0]])
                for gt in gt_list_int:
                    if gt[0] < image.size[0] and gt[1] < image.size[1]:
                        gt_matrix[gt[1], gt[0]] = 1.
                # gaussian_filter_density(gt_matrix, dens_path, index)
                pool.apply_async(gaussian_filter_density, (gt_matrix, dens_path, index,))
            pool.close()
            pool.join()

    for part_index, set_part in enumerate(shtech_set_path):  # 生成tfrecord文件
        for class_index, set_class in enumerate(set_part):
            record_path = os.path.join(set_class, 'all_data.tfrecords')
            image_paths = shtech_image_path[part_index][class_index]
            writer = tf.python_io.TFRecordWriter(record_path)  # tfrecords的写法
            for img_p in image_paths:
                dens_p = img_p.replace('images', 'dens_np').replace('IMG', 'numpy').replace('.jpg', '.npy')
                img_file = PIL.Image.open(img_p)
                dens_file = np.load(dens_p)
                img_file = img_file.convert('RGB')  # 要将黑白图片变成三通道
                img_array, dens_array = np.array(img_file), np.array(dens_file)
                rand_times = 5
                img_croplist, dens_croplist, height, widht = crop_function(img_array, dens_array, rand_times)
                for file_index in range(4+rand_times):
                    img_array_temp = img_croplist[file_index]
                    dens_array_temp = dens_croplist[file_index]
                    img_raw_temp = img_array_temp.tostring()
                    dens_raw_temp = dens_array_temp.tostring()
                    feature = {
                        'height': int64_feature(height),
                        'width': int64_feature(widht),
                        'img': bytes_feature(img_raw_temp),
                        'dens': bytes_feature(dens_raw_temp)
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            writer.close()
            print(set_class)
