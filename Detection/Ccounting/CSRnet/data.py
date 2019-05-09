import numpy as np
from matplotlib import pyplot as plt
import scipy
import os
import glob


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[0].flatten(), np.nonzero(gt)[1].flatten())))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


root = 'Datasets/shtech'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    mat = scipy.io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground')
    with h5py.File(file_path, 'w') as hf:
        hf['density'] = k
