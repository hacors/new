import glob
import multiprocessing as multp
import os

import numpy as np
import PIL
import scipy
import scipy.io as scio
import scipy.ndimage as scnd
from matplotlib import cm
from matplotlib import pyplot as plt

ROOT = 'Datasets'


def gaussian_filter_density(ground_truth, density_path, index):
    density = np.zeros(ground_truth.shape, dtype=np.float32)
    positions = np.array(list(zip(np.nonzero(ground_truth)[0].ravel(), np.nonzero(ground_truth)[1].ravel())))
    tree = scipy.spatial.KDTree(positions.copy(), leafsize=2048)
    distances, locations = tree.query(positions, k=4)
    if len(positions) == 1:
        temp_filter = np.zeros(ground_truth.shape, dtype=np.float32)
        temp_filter[positions[0][0], positions[0][1]] = 1.
        sigma = np.average(np.array(ground_truth.shape))/4.
        density += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    else:
        for index, position in enumerate(positions):
            temp_filter = np.zeros(ground_truth.shape, dtype=np.float32)
            temp_filter[position[0], position[1]] = 1.
            sigma = (distances[index][1]+distances[index][2]+distances[index][3])*0.1  # 这考虑了一种透视折衷
            density += scnd.filters.gaussian_filter(temp_filter, sigma, mode='constant')
    print('do', index)
    np.savetxt(density_path, density)


def get_shtech_path(root=ROOT):
    root = os.path.join(root, 'shtech')
    part_A_train = os.path.join(root, 'part_A_final', 'train_data')
    part_A_test = os.path.join(root, 'part_A_final', 'test_data')
    part_B_train = os.path.join(root, 'part_B_final', 'train_data')
    part_B_test = os.path.join(root, 'part_B_final', 'test_data')
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


def show_image(all_image_path):
    for image_path in all_image_path:
        true_image = PIL.Image.open(image_path)
        plt.imshow(true_image)
        plt.show()
        density_path = image_path.replace('.jpg', '.txt').replace('images', 'ground')
        ground_truth = np.loadtxt(density_path)
        plt.imshow(ground_truth, cmap=cm.jet)
        plt.show()


if __name__ == "__main__":
    all_image_path, all_gt_path = get_shtech_path()
    # show_image(all_image_path[:100])
    all_ground_truth = list()
    all_density_path = list()
    nums = len(all_image_path)
    for index in range(nums):
        mat = scio.loadmat(all_gt_path[index])
        gt_list = mat['image_info'][0, 0][0, 0][0]  # 注意gt的坐标是笛卡尔坐标
        gt_list_int = gt_list.astype(np.int16)
        image = plt.imread(all_image_path[index])
        ground_truth = np.zeros(image.shape[:2])
        for gt in gt_list_int:
            if gt[1] < image.shape[0] and gt[0] < image.shape[1]:
                ground_truth[gt[1], gt[0]] = 1
        density_path = all_image_path[index].replace('.jpg', '.txt').replace('images', 'ground')
        all_ground_truth.append(ground_truth)
        all_density_path.append(density_path)
    pool = multp.Pool(processes=12)
    for index in range(nums):
        pool.apply_async(gaussian_filter_density, (all_ground_truth[index], all_density_path[index], index,))
    pool.close()
    pool.join()
