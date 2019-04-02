import numpy as np
import sklearn
from matplotlib import pyplot as plt
from matplotlib import colors

txtdirector = 'Practice/Bills/Active/data_banknote_authentication.txt'
SVC = sklearn.svm.LinearSVC


def getdata():
    file_object = open(txtdirector)
    file_content = file_object.read().splitlines()
    datas, targets = [], []
    for line in file_content:
        line = line.split(',')
        datas.append(list(map(float, line[:-1])))
        targets.append(int(line[-1]))
    datas = np.array(datas)
    targets = np.array(targets)
    return(datas, targets)


def show_scatter(datas, targets):
    plt.scatter(datas[targets == 0, 0], datas[targets == 0, 1], color='red')
    plt.scatter(datas[targets == 1, 0], datas[targets == 1, 1], color='blue')
    plt.show()


def show_boundary(model, datas, targets):
    x_min, x_max = datas[:, 0].min()-0.5, datas[:, 0].max()+0.5
    y_min, y_max = datas[:, 1].min()-0.5, datas[:, 1].max()+0.5
    step = 0.01
    x_matrix, y_matrix = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    coordinate = np.c_(x_matrix.ravel(), y_matrix.ravel())
    predict = model.predict(coordinate).reshape(x_matrix.shape)
    custom = colors.ListedColormap(['red', 'blue', 'black'])
    plt.contourf(x_matrix, y_matrix, predict, cmap=custom)
    show_scatter(datas, targets)


datas, targets = getdata()
show_scatter(datas, targets)
simplesvc = SVC(penalty='L1', C=1e9)
simplesvc.fit(datas, targets)
show_boundary(simplesvc, datas, targets)
