from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
datas_list = [[178, 75], [163, 56], [163, 50], [168, 49], [175, 70], [160, 45], [160, 50],
              [156, 38], [173, 60], [165, 60], [174, 65], [176, 58], [170, 80], [153, 42],
              [162, 60], [171, 67], [161, 45], [165, 60], [163, 60], [172, 60], [165, 45],
              [173, 54], [159, 43], [158, 52], [164, 55], [169, 60], [175, 60], [180, 71]]
labels_list = ['male', 'female', 'female', 'female', 'male', 'female', 'female',
               'female', 'male', 'female', 'male', 'male', 'male', 'female',
               'female', 'male', 'female', 'male', 'male', 'male', 'female',
               'male', 'female', 'female', 'female', 'male', 'male', 'male']

my_classifier = knn(n_neighbors=6)
my_classifier.fit(datas_list, labels_list)
test = np.array([[180, 63], [178, 70], [175, 65], [172, 58], [170, 52], [169, 57], [168, 65],
                 [158, 50], [160, 48], [164, 54], [166, 59], [169, 58], [170, 48], [172, 55]])
                 # 这部分为测试，可自行添加
print(my_classifier.predict(test))
