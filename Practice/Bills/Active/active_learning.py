import random

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing

txtdirector = 'Practice/Bills/Active/data_banknote_authentication.txt'
C_grid = [{'C': list(10**i for i in range(-6, 6))}]


def getdata():
    file_object = open(txtdirector)
    file_content = file_object.read().splitlines()
    p_datas, p_targets = [], []
    for line in file_content:
        line = line.split(',')
        p_datas.append(list(map(float, line[:-1])))
        p_targets.append(int(line[-1]))
    p_datas = np.array(p_datas)
    p_targets = np.array(p_targets)
    standarder = preprocessing.StandardScaler()
    p_datas = standarder.fit_transform(p_datas)
    return(p_datas, p_targets)


def get_train_test(p_datas, p_targets):
    indexlist = list(range(1372))
    random.shuffle(indexlist)
    train_data, train_target = p_datas[indexlist[:900]], p_targets[indexlist[:900]]
    test_data, test_target = p_datas[indexlist[900:]], p_targets[indexlist[900:]]
    return (train_data, train_target), (test_data, test_target)


def show_scatter(p_datas, p_targets):
    plt.scatter(p_datas[p_targets == 0, 0], p_datas[p_targets == 0, 1], color='red')
    plt.scatter(p_datas[p_targets == 1, 0], p_datas[p_targets == 1, 1], color='blue')
    plt.show()


def show_boundary(model, p_datas, p_targets):
    x_min, x_max = p_datas[:, 0].min()-0.5, p_datas[:, 0].max()+0.5
    y_min, y_max = p_datas[:, 1].min()-0.5, p_datas[:, 1].max()+0.5
    step = 0.01
    x_matrix, y_matrix = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    coordinate = np.c_[x_matrix.ravel(), y_matrix.ravel()]
    predict = model.predict(coordinate).reshape(x_matrix.shape)
    custom = colors.ListedColormap(['orange', 'black', 'green'])
    plt.contourf(x_matrix, y_matrix, predict, cmap=custom)
    show_scatter(p_datas, p_targets)


def passive_learning(p_train_datas, p_train_targets, p_test_datas, p_test_targets):
    accuracy = list()
    for repeat in range(50):
        for i in range(90):
            rangeofdata = 10*(i+1)
            datas = p_train_datas[:rangeofdata]
            targets = p_train_targets[:rangeofdata]
            para_C = get_grided_para(datas, targets)
            tempsvc = svm.LinearSVC(penalty='l1', dual=False, C=para_C)
            tempsvc.fit(datas, targets)
            predicts = tempsvc.predict(p_test_datas)
            accu = sum(predicts == p_test_targets)/472
            accuracy.append(accu)
            print(accu)
    return(accuracy)


def get_grided_para(p_datas, p_targets):
    grid_svc = svm.LinearSVC(penalty='l1', dual=False)
    if len(p_targets) == 10:
        cv_num = 5
    else:
        cv_num = 10
    grid_search = model_selection.GridSearchCV(grid_svc, C_grid, n_jobs=-1, cv=cv_num, iid=True)
    grid_search.fit(p_datas, p_targets)
    return(grid_search.best_params_['C'])


datas, targets = getdata()
# datas = datas[:, :2]
(train_data, train_target), (test_data, test_target) = get_train_test(datas, targets)
accu = passive_learning(train_data, train_target, test_data, test_target)
print(accu)
