import scipy.io as scio

mat = scio.loadmat('Datasets/Sh_tech/part_A_final/train_data/ground_truth/GT_IMG_52.mat')
gt = mat["image_info"][0, 0][0, 0]
print(gt.T.T.T.T.T.T.T)
print(type(gt))
other = gt.T
print(1)
