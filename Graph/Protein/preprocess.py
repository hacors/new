import utils
temp_path = 'Datasets/Protien/ess_info/ess_archaea.dat'


def get_ess_gene(path):
    temp = open(path)
    for row in temp:
        temp = row.strip().split('\t')
        print(temp)
    pass


get_ess_gene(temp_path)
