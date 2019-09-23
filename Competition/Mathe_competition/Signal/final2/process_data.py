import pandas as pd
import numpy as np
import os


def my_process(path):
    data = pd.read_csv(path)
    # change below
    data['Height'] = np.where(data['Height'] == 0, 22, data['Height'])
    data['downtilt'] = data['Electrical Downtilt'] + data['Mechanical Downtilt']
    data['h_theta'] = 90 - np.rad2deg(np.arctan2(data['Y'] - data['Cell Y'], data['X'] - data['Cell X'])) - data[
        'Azimuth']
    data['h_theta'] = np.where(data['h_theta'] > 180, data['h_theta'] - 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] > 180, data['h_theta'] - 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] < -180, data['h_theta'] + 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] < -180, data['h_theta'] + 360, data['h_theta'])
    data['distance'] = ((data['X'] - data['Cell X']) ** 2 + (data['Y'] - data['Cell Y']) ** 2) ** 0.5
    data['delta_altitude'] = data['Altitude'] - data['Cell Altitude']
    data['deltaHv'] = data['Height'] - np.tan(np.deg2rad(data['downtilt'])) * data['distance'] * np.cos(
        np.deg2rad(data['h_theta']))
    data['direct_distance'] = ((np.abs(data['deltaHv']) * np.cos(np.deg2rad(data['downtilt']))) ** 2 + (
        data['distance'] * np.sin(np.deg2rad(data['h_theta']))) ** 2) ** 0.5
    data['pass_distance'] = (np.abs(
        data['deltaHv'] ** 2 + data['distance'] ** 2 - data['direct_distance'] ** 2)) ** 0.5
    data['Height'] = np.log10(data['Height'])
    data['Frequency Band'] = np.log10(data['Frequency Band'])
    data['distance'] = np.log10(data['distance'] + 1)
    data['height_mul_distance'] = data['Height'] * data['distance']
    data['direct_distance'] = np.log10(data['direct_distance'] + 1)
    data['pass_distance'] = np.log10(data['pass_distance'] + 1)
    data['Cell Building Height'] = np.where(data['Cell Building Height'] > 0, 1, 0)
    data['Building Height'] = np.where(data['Building Height'] > 0, 1, 0)
    print(data.shape)
    data = data.drop(
        ['Frequency Band', 'Cell Building Height', 'Cell Clutter Index', 'Clutter Index', 'Building Height',
            'RS Power', 'Cell Index', 'Cell X', 'Cell Y', 'Azimuth', 'Electrical Downtilt', 'Mechanical Downtilt', 'X',
            'Y', 'Altitude', 'Cell Altitude', 'downtilt'], axis=1)
    return data


def cut_and_store(data, director):
    data = data.sample(frac=1.0)
    length = data.shape[0]//8192
    for index in range(length):
        temp_data = data[index*8192:(index+1)*8192]
        path = os.path.join(director, str(index)+'.csv')
        temp_data.to_csv(path, index=None)  # 只能使用index=None去除index


if __name__ == '__main__':
    data = my_process('Datasets/huawei_signal/train_processed_set/train_merge.csv')
    cut_and_store(data, 'Datasets/huawei_signal/train_processed_set/cuted_data')
