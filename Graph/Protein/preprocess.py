import utils
from collections import Counter
import pandas as pd
datasets = ['ess_eukaryotes', 'ess_archaea', 'ess_bacteria']
temp_path = 'Datasets/Protein/ess_info/ess_eukaryotes.dat'
temp_path = 'Datasets/Protein/ess_info/ess_archaea.dat'
temp_path = 'Datasets/Protein/ess_info/ess_bacteria.dat'


def get_organ_gene(path):
    files = open(path)
    temp_file = []
    # col_heads = ['deg_id_0', 'deg_id_1', 'gene_name', 'gene_ref', 'cog', 'class', 'func', 'organism', 'res_seq', 'condi', 'append_0', 'append_1', 'append_2', 'append_3']
    for row in files:
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
    all_result = Counter()
    for setname in datasets:
        temp_path = 'Datasets/Protein/ess_info/%s.dat' % setname
        all_result = all_result+Counter(get_organ_gene(temp_path))
    return dict(all_result)


result = get_organ_gene(temp_path)

all_result = get_all_data()
pass
