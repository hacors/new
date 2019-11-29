import pandas as pd
import os
sum_info = []
result = set()
for path in os.listdir('Data/Temp'):
    temp_path = 'Data/Temp/%s' % path
    temp_info = []
    for row in open(temp_path):
        list_info = row.strip().split('\t')
        temp_info.append(list_info)
    temp_info.pop(0)
    for temp in temp_info:
        choose_1, choose_2 = temp[7], temp[8]
        if choose_1[:3] == 'NC_':
            result.add(choose_2)
        else:
            result.add(choose_1)
    pass
print(result)
print(result.__len__())
'''
datasets = ['eukaryotes', 'archaea', 'bacteria']

def read_file(path):
    temp_file = []
    # col_heads = ['deg_id_0', 'deg_id_1', 'gene_name', 'gene_ref', 'cog', 'class', 'func', 'organism', 'res_seq', 'condi', 'append_0', 'append_1', 'append_2', 'append_3']
    for row in open(path):
        temp_file.append(row.strip().split('\t'))


def get_all_data():
    all_result = {}
    for setname in datasets:
        temp_path = 'Datasets/Protein/Data/Temp/%s.dat' % setname
        temp_result = read_file(temp_path)
        for ogran_name in temp_result:
            all_result[ogran_name] = temp_result[ogran_name]
    return all_result


result = read_file(temp_path)

all_result = get_all_data()
'''
