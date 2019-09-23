import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd

import pickle


class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        file_data = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                file_data.append(input_data)

        file_data = np.array(file_data, dtype=np.float32).reshape(-1, 17)

        preprocessed_data['myInput'] = file_data
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        print(self.model_path)

        output = self.my_process(preprocessed_data)
        print("yubzhu successfully insert data to model!")
        return output

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output

    def my_process(self, data):
        data = pd.DataFrame(data['myInput'])
        data.columns = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
                        'Mechanical Downtilt',
                        'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height', 'Cell Clutter Index',
                        'X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']
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
        print(data.shape)
        output = np.array(data, dtype=np.float32).reshape(-1, 8)
        output_dict = {}
        output_dict['myInput'] = output
        return output_dict


'''
        print(data.shape)
        fr = open(self.model_path + '/misc/model', 'rb')
        model = pickle.load(fr)
        output = model.predict(data)
        output = np.array(output, dtype=np.float32).reshape(-1, 1)
        print(output.shape)
        output_dict = {}
        output_dict['myInput'] = output
        return output_dict
'''
