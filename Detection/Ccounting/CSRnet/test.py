import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
file_path = 'Datasets/shtech/part_A_final/train_data/ground/IMG_10.txt'
ground_truth = np.loadtxt(file_path)
plt.imshow(ground_truth, cmap=cm.jet)
plt.show()
