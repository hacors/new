import utils
import pandas as pd

datasets = ['eukaryotes', 'archaea', 'bacteria']

'''
def read_file(path):
    temp_file = []
    # col_heads = ['deg_id_0', 'deg_id_1', 'gene_name', 'gene_ref', 'cog', 'class', 'func', 'organism', 'res_seq', 'condi', 'append_0', 'append_1', 'append_2', 'append_3']
    for row in open(path):
        temp_file.append(row.strip().split('\t'))
    temp_file.pop(0)
    df_file = pd.DataFrame(temp_file)
    # print(df_file.head(10))
    df_file_group = df_file.groupby(7)
    organ_result = {}
    for organ_name, organ_data in df_file_group:
        organ_gene = list(organ_data[2])
        if not organ_gene.count('-'):  # 去除'-'
            organ_gene = [gene for temp_str in organ_gene for gene in temp_str.split('/')]  # 将'/'一分为二
            organ_result[organ_name] = organ_gene
    return organ_result


def get_all_data():
    all_result = {}
    for setname in datasets:
        temp_path = 'Datasets/Protein/ess_info/%s.dat' % setname
        temp_result = read_file(temp_path)
        for ogran_name in temp_result:
            all_result[ogran_name] = temp_result[ogran_name]
    return all_result


result = read_file(temp_path)

all_result = get_all_data()
pass
'''
