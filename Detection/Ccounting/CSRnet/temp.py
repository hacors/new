import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
file_path = 'Datasets/shtech/part_A_final/train_data/ground/IMG_10.h5'
gt_file = h5py.File(file_path, 'r')
groundtruth = np.asarray(gt_file['density'])
groundtruth = groundtruth/groundtruth.max()
plt.imshow(groundtruth,CM.jet)
plt.show()
print(1)
