import pandas as pd
import numpy as np


def my_process(file_content):
    data = pd.read_csv(file_content)
    data = np.array(data.get_values()[:, 0:18], dtype=np.float32)
    data = pd.DataFrame(data)
    data.columns = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
                    'Mechanical Downtilt',
                    'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height', 'Cell Clutter Index',
                    'X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']
    data = data[data['Height'] != 0]
    # 垂直下倾角
    data['downtilt'] = data['Electrical Downtilt'] + data['Mechanical Downtilt']
    # 水平差角
    data['h_theta'] = 90 - np.rad2deg(np.arctan2(data['Y'] - data['Cell Y'], data['X'] - data['Cell X'])) - data[
        'Azimuth']
    data['h_theta'] = np.where(data['h_theta'] > 180, data['h_theta'] - 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] > 180, data['h_theta'] - 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] < -180, data['h_theta'] + 360, data['h_theta'])
    data['h_theta'] = np.where(data['h_theta'] < -180, data['h_theta'] + 360, data['h_theta'])
    # data = data[np.abs(data['h_theta']) < 90]
    data['distance'] = ((data['X'] - data['Cell X']) ** 2 + (data['Y'] - data['Cell Y']) ** 2) ** 0.5
    data = data[data['distance'] * np.cos(np.deg2rad(data['h_theta'])) <= data['Height'] / np.tan(
        np.deg2rad(data['downtilt'])) * 1.5]
    # data = data[data['distance'] * np.cos(np.deg2rad(data['h_theta'])) <= 500]
    # 相对海拔高度
    data['delta_altitude'] = data['Altitude'] - data['Cell Altitude']
    # deltaHv
    data['deltaHv'] = data['Height'] - np.tan(np.deg2rad(data['downtilt'])) * data['distance'] * np.cos(
        np.deg2rad(data['h_theta']))
    # 垂线距离
    data['direct_distance'] = ((np.abs(data['deltaHv']) * np.cos(np.deg2rad(data['downtilt']))) ** 2 + (
        data['distance'] * np.sin(np.deg2rad(data['h_theta']))) ** 2) ** 0.5
    # 传播距离
    data['pass_distance'] = (np.abs(
        data['deltaHv'] ** 2 + data['distance'] ** 2 - data['direct_distance'] ** 2)) ** 0.5
    data['Height'] = np.log10(data['Height'])

    data['Frequency Band'] = np.log10(data['Frequency Band'])

    data = data[data['distance'] != 0]
    data['distance'] = np.log10(data['distance'])

    data = data[data['direct_distance'] != 0]
    data['direct_distance'] = np.log10(data['direct_distance'])

    data = data[data['pass_distance'] != 0]
    data['pass_distance'] = np.log10(data['pass_distance'])

    data = data.drop(
        ['Frequency Band', 'RS Power', 'Cell Index', 'Cell X', 'Cell Y', 'Azimuth', 'Electrical Downtilt',
         'Mechanical Downtilt', 'X', 'Y', 'Altitude', 'Cell Altitude', 'h_theta', 'downtilt'], axis=1)

    data['Cell Building Height'] = np.where(data['Cell Building Height'] > 0, 1, 0)
    data['Building Height'] = np.where(data['Building Height'] > 0, 1, 0)
    return data
