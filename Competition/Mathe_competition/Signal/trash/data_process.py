import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
TRAIN_ROOT = r'D:\code\Datasets\huawei_data\train_set'
TEST_ROOT = r'D:\code\Datasets\huawei_data\test_set'
PROCESSED_ROOT = r'D:\code\Datasets\huawei_data\train_processed_set'


def merge_train_set(t_root, m_dir):  # 给出文件和合并之后的目标文件
    if os.path.exists(m_dir):
        os.remove(m_dir)
    with open(m_dir, 'a', newline='') as merge_file:
        merge_writer = csv.writer(merge_file)
        first_file = os.listdir(t_root)[0]
        title = list(csv.reader(open(os.path.join(t_root, first_file), 'r')))[0]
        merge_writer.writerow(title)
        for file_path in os.listdir(t_root):
            with open(os.path.join(t_root, file_path), 'r') as temp_file:
                file_data = list(csv.reader(temp_file))  # reader获取读取器
                merge_writer = csv.writer(merge_file)  # writer获取书写器
                merge_writer.writerows(file_data[1:])  # 书写器可以一次写多行


def simple_process(m_dir):
    # 简单的对index计数
    # [0, 157233, 0, 0, 6159532, 2397951, 739713, 108659, 0, 268933, 151335, 718328, 776184, 368303, 135082, 12374, 13714, 4492, 0, 0]
    count_of_label = [0 for label in range(20)]
    with open(m_dir, 'r') as merge_file:
        file = csv.reader(merge_file)
        for row in file:
            count_of_label[int(row[16])-1] += 1
    print(count_of_label)


def draw(input_data):
    x_index = list(range(len(input_data)))
    plt.bar(x_index, input_data)
    plt.show()


def process_of_building(data):
    X_min, Y_min = min(data['X']), min(data['Y'])
    X_max, Y_max = max(data['X']), max(data['Y'])
    Altitude_base = min(min(data['Altitude']), min(data['Cell Altitude']))
    data['_angle_level'] = np.arctan2(data['X']-data['Cell X'], data['Y']-data['Cell Y'])*180/np.pi
    data['_relative_x'] = data['X']-data['Cell X']
    data['_relative_y'] = data['Y']-data['Cell Y']
    data['_distance'] = (data['_relative_x']**2 + data['_relative_y']**2)**0.5
    data['occlusion'] = 0
    tour_pos_Y = int(data['Cell Y'][0]-Y_min)//5
    tour_pos_X = int(data['Cell X'][0]-X_min)//5
    tour_height = int(data['Cell Altitude'][0])-Altitude_base
    shape = (int((Y_max-Y_min)/5+1), int((X_max-X_min)/5+1))
    geo_array = np.zeros(shape, dtype=np.int)
    for index, line in data.iterrows():
        temp_pos_Y = int(line['Y']-Y_min)//5
        temp_pos_X = int(line['X']-X_min)//5
        geo_array[temp_pos_Y][temp_pos_X] = line['Altitude']-Altitude_base
    for index, line in data.iterrows():
        target_pos_Y = int(line['Y']-Y_min)//5
        target_pos_X = int(line['X']-X_min)//5
        target_height = geo_array[target_pos_Y][target_pos_X]
        theta = line['_angle_level']
        remenber_Y = tour_pos_Y
        remenber_X = tour_pos_X
        result = 0
        for dis_step in range(max((int(line['_distance'])//5-10), int(line['_distance'])//10), int(line['_distance'])//5):
            distance_in_grid = dis_step
            Y = int(tour_pos_Y+distance_in_grid*np.cos(theta))
            X = int(tour_pos_X+distance_in_grid*np.sin(theta))
            if (Y != remenber_Y or X != remenber_X) and Y >= 0 and X >= 0 and Y < shape[0] and X < shape[1] and geo_array[Y][X]:
                # do something in result
                if geo_array[Y][X] >= (tour_height*((line['_distance'])-dis_step)+target_height*dis_step)/(line['_distance']):
                    result += 1
                remenber_Y = Y
                remenber_X = X
        data.loc[index, 'occlusion'] = result
    return data[['occlusion']]


def merge_lie(t_root, m_dir):
    if os.path.exists(m_dir):
        os.remove(m_dir)
    pd_file = pd.read_csv(os.path.join(t_root, os.listdir(t_root)[0]))
    result = process_of_building(pd_file)
    for index, file_path in enumerate(os.listdir(t_root)[1:]):
        print(index)
        pd_file = pd.read_csv(os.path.join(t_root, file_path))
        needed_data = process_of_building(pd_file)
        result = result.append(needed_data)
    result.to_csv(m_dir)


def get_pos_of_signal():
    pos_list = []
    for file_path in os.listdir(TRAIN_ROOT):
        pd_file = pd.read_csv(os.path.join(TRAIN_ROOT, file_path))
        pos_list.append([pd_file['Cell X'][0], pd_file['Cell Y'][0]])
    np.savetxt(os.path.join(PROCESSED_ROOT, 'pos_list.txt'), np.array(pos_list))
    pos = np.loadtxt(os.path.join(PROCESSED_ROOT, 'pos_list.txt'), dtype=np.int32)
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


def get_lists(root):
    all_list = []
    for index, file_path in enumerate(os.listdir(root)):
        data = pd.read_csv(os.path.join(root, file_path))
        X_min, Y_min = min(data['X']), min(data['Y'])
        X_max, Y_max = max(data['X']), max(data['Y'])
        shape = (int((Y_max-Y_min)/5+1), int((X_max-X_min)/5+1))
        geo_array = np.ones(shape, dtype=np.int)
        geo_array = geo_array*(-1)
        for index, line in data.iterrows():
            temp_pos_Y = int(line['Y']-Y_min)//5
            temp_pos_X = int(line['X']-X_min)//5
            geo_array[temp_pos_Y][temp_pos_X] = int(line['Cell Clutter Index'])
        begin = []
        for index, line in data.iterrows():
            temp_pos_Y = int(line['Y']-Y_min)//5
            temp_pos_X = int(line['X']-X_min)//5
            for nei_y in range(temp_pos_Y-1, temp_pos_Y+1):
                for nei_x in range(temp_pos_X-1, temp_pos_X+1):
                    if geo_array[nei_y][nei_x] != -1:
                        begin.append(str(geo_array[nei_y][nei_x]))
        all_list.append(begin)
    return all_list


if __name__ == '__main__':
    merge_director = os.path.join(PROCESSED_ROOT, 'train_merge.csv')
    toy_director = os.path.join(PROCESSED_ROOT, 'train_toy.csv')
    occlusion_director = os.path.join(PROCESSED_ROOT, 'occlusion.csv')
    # merge_train_set(TRAIN_ROOT, merge_director)
    # simple_process(merge_director)
    # merge_data = pd.read_csv(merge_director)
    # the_datas = pd.read_csv(toy_director)
    # process_of_building(the_datas)
    # print(the_datas.head)
    # merge_lie(TRAIN_ROOT, occlusion_director)
    # get_lists(TRAIN_ROOT)
