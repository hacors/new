import glob
import multiprocessing as multp
import os

import h5py
import numpy as np
import PIL
import scipy
import scipy.io as scio
import scipy.ndimage as scnd
from matplotlib import cm
from matplotlib import pyplot as plt

ROOT = 'Datasets'


def gaussian_filter_density(p_gt_matrix, p_dens_path, p_index):
    dens_image = np.zeros(p_gt_matrix.shape, dtype=np.float32)
    pos_list = np.array(list(zip(np.nonzero(p_gt_matrix)[0].flatten(), np.nonzero(p_gt_matrix)[1].flatten())))
    kd_tree = scipy.spatial.KDTree(pos_list.copy(), leafsize=2048)
    kd_dis, kd_locat = kd_tree.query(pos_list, k=4)
    if len(pos_list) == 1:
        temp_filter = np.zeros(p_gt_matrix.shape, dtype=np.float32)
        temp_filter[pos_list[0][0], pos_list[0][1]] = 1.
        sigma = np.average(np.array(p_gt_matrix.shape))/4.
        dens_image += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    else:
        for i, pos in enumerate(pos_list):
            temp_filter = np.zeros(p_gt_matrix.shape, dtype=np.float32)
            temp_filter[pos[0], pos[1]] = 1.
            sigma = (kd_dis[i][1]+kd_dis[i][2]+kd_dis[i][3])*0.1  # 这考虑了一种透视折衷
            dens_image += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    with h5py.File(p_dens_path, 'w') as hf:
        hf['density'] = dens_image
    print('finish:', p_index)


def get_shtech_path():
    temp_root = os.path.join(ROOT, 'shtech')
    part_A_train = os.path.join(temp_root, 'part_A_final', 'train_data')
    part_A_test = os.path.join(temp_root, 'part_A_final', 'test_data')
    part_B_train = os.path.join(temp_root, 'part_B_final', 'train_data')
    part_B_test = os.path.join(temp_root, 'part_B_final', 'test_data')
    path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]
    all_image_path = []
    all_gt_path = []
    for path in path_sets:
        try:
            os.mkdir(os.path.join(path, 'ground'))
        except Exception:
            pass
        for ima_path in glob.glob(os.path.join(path, 'images', '*.jpg')):
            all_image_path.append(ima_path)
        for gt_path in glob.glob(os.path.join(path, 'ground_truth', '*.mat')):
            all_gt_path.append(gt_path)
    return all_image_path, all_gt_path


def show_image(p_imapath_list):
    for image_path in p_imapath_list:
        image = PIL.Image.open(image_path)
        plt.imshow(image)
        plt.show()
        dens_path = image_path.replace('.jpg', '.h5').replace('images', 'ground')
        dens_file = h5py.File((dens_path), 'r')
        dens_image = np.asarray(dens_file['density'])
        plt.imshow(dens_image, cmap=cm.jet)
        plt.show()


if __name__ == "__main__":
    all_image_path, all_gt_path = get_shtech_path()
    # show_image(all_image_path[5:7])
    pool = multp.Pool(processes=12)
    for index in range(len(all_image_path)):
        mat = scio.loadmat(all_gt_path[index])
        gt_list = mat['image_info'][0, 0][0, 0][0]  # 注意gt的坐标是笛卡尔坐标
        gt_list_int = gt_list.astype(np.int)
        image = plt.imread(all_image_path[index])
        gt_matrix = np.zeros(image.shape[:2])
        for gt in gt_list_int:
            if gt[1] < image.shape[0] and gt[0] < image.shape[1]:
                gt_matrix[gt[1], gt[0]] = 1
        dens_path = all_image_path[index].replace('.jpg', '.h5').replace('images', 'ground')
        pool.apply_async(gaussian_filter_density, (gt_matrix, dens_path, index,))
    pool.close()
    pool.join()
