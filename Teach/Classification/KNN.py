import numpy as np
from matplotlib import pyplot as plt

datas_list = [[178, 75], [163, 56], [163, 50], [168, 49], [175, 70], [160, 45], [160, 50],
              [156, 38], [173, 60], [165, 60], [174, 65], [176, 58], [170, 80], [153, 42],
              [162, 60], [171, 67], [161, 45], [165, 60], [163, 60], [172, 60], [165, 45],
              [173, 54], [159, 43], [158, 52], [164, 55], [169, 60], [175, 60], [180, 71]]
labels_list = ['male', 'female', 'female', 'female', 'male', 'female', 'female',
               'female', 'male', 'female', 'male', 'male', 'male', 'female',
               'female', 'male', 'female', 'male', 'male', 'male', 'female',
               'male', 'female', 'female', 'female', 'male', 'male', 'male']
datas_array = np.array(datas_list)
labels_array = np.array(labels_list)

plt.scatter(datas_array[labels_array == 'male', 0], datas_array[labels_array == 'male', 1], color='blue', label='male')
plt.scatter(datas_array[labels_array == 'female', 0],
            datas_array[labels_array == 'female', 1], color='red', label='female')
plt.scatter(169, 57, color='green', label='test')
plt.legend()
plt.show()
