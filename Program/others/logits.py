import numpy as np
np.random.seed(666)
# 网络中一共4层10,8,5,1,以此为第0,1,2,3层,其中10-8为第一层全连接
net_struc = [{'input_dims': 10, 'output_dims': 8, 'activation': 'relu'},
             {'input_dims': 8, 'output_dims': 5, 'activation': 'relu'},
             {'input_dims': 5, 'output_dims': 1, 'activation': 'sigmoid'}
             ]


def init_layers(struct_list):
    params_values = {}
    for index, struct in enumerate(struct_list):
        layer_index = index+1
        layer_input_size = struct['input_dims']
        layer_output_size = struct['output_dims']
        

    return params_values


if __name__ == '__main__':
    init_layers(net_struc)


'''
def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values
'''
