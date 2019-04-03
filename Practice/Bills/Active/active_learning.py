import heapq
import math
import random
from collections import Counter

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn import model_selection, svm

# from sklearn import preprocessing

# txtdirector = 'Practice/Bills/Active/data_banknote_authentication.txt'
# save_passive = 'Practice/Bills/Active/save_passive.txt'
# save_active = 'Practice/Bills/Active/save_active.txt'
txtdirector = 'data_banknote_authentication.txt''
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
    return(p_datas, p_targets)


def get_train_test(p_datas, p_targets):
    shuffled_datas, shuffled_targets = do_shuffle(p_datas, p_targets)
    train_data, train_target = shuffled_datas[:900], shuffled_targets[:900]
    test_data, test_target = shuffled_datas[900:], shuffled_targets[900:]
    return (train_data, train_target), (test_data, test_target)


def do_shuffle(p_datas, p_targets):
    indexlist = list(range(len(p_datas)))
    success = False
    while not success:
        random.shuffle(indexlist)
        shuffled_datas = p_datas[indexlist]
        shuffled_targets = p_targets[indexlist]
        check = shuffled_targets[:10]
        count = Counter(check)
        if count[0] >= 3 and count[1] >= 3:
            success = True
    return(shuffled_datas, shuffled_targets)


def get_nearst(coef, intercept, p_datas, p_targets):
    if len(p_datas) == 0:
        return(p_datas, p_targets, p_datas, p_targets)
    else:
        selected_datas = list()
        selected_targets = list()
        rest_datas = list(p_datas.copy())
        rest_targets = list(p_targets.copy())
        distance_list = list()
        for data in p_datas:
            distance_list.append(get_distance(coef, intercept, data))
        smallest_index = list(map(distance_list.index, heapq.nsmallest(10, distance_list)))
        smallest_index.sort(key=None, reverse=True)
        for index in smallest_index:
            selected_datas.append(p_datas[index])
            selected_targets.append(p_targets[index])
            rest_datas.pop(index)
            rest_targets.pop(index)
        return(np.array(selected_datas), np.array(selected_targets), np.array(rest_datas), np.array(rest_targets))


def get_distance(coef, intercept, data):
    sums = math.sqrt(np.sum(coef**2))
    '''
    dis_a = abs(np.sum(coef*data)+intercept-1)/sums
    dis_b = abs(np.sum(coef*data)+intercept+1)/sums
    return min([dis_a, dis_b])
    '''
    return abs(np.sum(coef*data)+intercept)/sums

'''
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
'''

def passive_learning(p_train_datas, p_train_targets, p_test_datas, p_test_targets):
    accuracy = list()
    for repeat in range(50):
        # print('passive', repeat)
        shuffled_train_datas, shuffled_train_targets = do_shuffle(p_train_datas, p_train_targets)
        for i in range(90):
            rangeofdata = 10*(i+1)
            pool_datas = shuffled_train_datas[:rangeofdata]
            pool_targets = shuffled_train_targets[:rangeofdata]
            tempsvc = get_grided_model(pool_datas, pool_targets)
            tempsvc.fit(pool_datas, pool_targets)
            predicts = tempsvc.predict(p_test_datas)
            accu = sum(predicts == p_test_targets)/472
            accuracy.append(accu)
            # print(accu, len())
    return(accuracy)


def active_learning(p_train_datas, p_train_targets, p_test_datas, p_test_targets):
    accuracy = list()
    for repeat in range(50):
        # print('active', repeat)
        shuffled_train_datas, shuffled_train_targets = do_shuffle(p_train_datas, p_train_targets)
        pool_datas, pool_targets = shuffled_train_datas[:10], shuffled_train_targets[:10]
        rest_datas, rest_targets = shuffled_train_datas[10:], shuffled_train_targets[10:]
        for i in range(90):
            tempsvc = get_grided_model(pool_datas, pool_targets)
            tempsvc.fit(pool_datas, pool_targets)
            predicts = tempsvc.predict(p_test_datas)
            accu = sum(predicts == p_test_targets)/472
            accuracy.append(accu)
            # print(accu, len(pool_datas))
            choosed_datas, choosed_targets, rest_datas, rest_targets = get_nearst(tempsvc.coef_[0], tempsvc.intercept_[0], rest_datas, rest_targets)
            if not len(choosed_datas) == 0:
                pool_datas = np.vstack((pool_datas, choosed_datas))
                pool_targets = np.hstack((pool_targets, choosed_targets))
    return(accuracy)


def get_grided_model(p_datas, p_targets):
    grid_svc = svm.LinearSVC(penalty='l1', dual=False, max_iter=20000, tol=1e-3)
    if len(p_targets) == 10:
        cv_num = 5
    else:
        cv_num = 10
    grid_search = model_selection.GridSearchCV(grid_svc, C_grid, n_jobs=-1, cv=cv_num, iid=True)
    grid_search.fit(p_datas, p_targets)
    return(grid_search.best_estimator_)


def run():
    datas, targets = getdata()
    (train_data, train_target), (test_data, test_target) = get_train_test(datas, targets)
    accu_passive = passive_learning(train_data, train_target, test_data, test_target)
    accu_active = active_learning(train_data, train_target, test_data, test_target)
    accu_passive = np.array(accu_passive).reshape(50, 90)
    accu_active = np.array(accu_active).reshape(50, 90)
    return accu_passive, accu_active
    # np.savetxt(save_passive, accu_passive)
    # np.savetxt(save_active, accu_active)


def show(accu_passive, acc_active):
    # accu_passive = np.loadtxt(save_passive)
    # acc_active = np.loadtxt(save_active)
    passive_list = np.mean(accu_passive, axis=0)
    active_list = np.mean(acc_active, axis=0)
    x_list = list(range(len(passive_list)))
    plt.scatter(x_list, passive_list, label='passive')
    plt.scatter(x_list, active_list, label='active')
    plt.legend()
    plt.show()


passive, acctive = run()
show(passive, acctive)
