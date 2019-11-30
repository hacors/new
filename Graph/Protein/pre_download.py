'''
在网上爬取的原始数据
从http://www.essentialgene.org/ 下载所有的关键蛋白信息
从https://www.ncbi.nlm.nih.gov/ 查找不同标准的生物的ID对应关系，存储于单独的文件夹
从https://string-db.org/ 下载所有相关生物的蛋白质相互作用信息，按照生物ID分类存储
从https://www.uniprot.org/ 查找所有相关的蛋白质注释信息，按照生物ID分类存储
'''
import os
import shutil
import urllib
import zipfile
import global_config
from fake_useragent import UserAgent
import requests
import random
import time
from bs4 import BeautifulSoup


def make_dir(director):
    director = director[:-1] if director[-1] == '/' else director
    if os.path.exists(director):
        shutil.rmtree(director)
    os.mkdir(director)


def internet_environment():  # 准备好爬取数据的网络环境
    uagent = UserAgent()
    chrome_uagent = uagent.data_browsers['chrome']
    headers = {'User-Agent': random.choice(chrome_uagent)}
    internet_env = dict()
    internet_env['headers'] = headers
    return internet_env


def main():
    print('Making directors and downloading......')
    # make_dir(global_config.DATA_ROOT)
    path_list = global_config.PATH_LIST
    # download_deg_files(path=path_list['DEG'])
    download_string_files(path=path_list['STRING'])
    download_uniport_files(path=path_list['UNIPORT'])
    print('Origin data is ready!')


def download_deg_files(path):
    make_dir(path)
    reaponse = requests.get('http://origin.tubic.org/deg/public/index.php/download')
    soup = BeautifulSoup(reaponse.text, features='lxml')
    links = soup.find('table').find_all('a')
    for index, link in enumerate(links[:3]):  # deg-p-15.2.zip,deg-p-15.2.zip,deg-e-15.2.zip
        down_link = 'http://origin.tubic.org'+link.attrs['href']
        down_name = global_config.DEG_BIO_CLASS[index]+'.zip'
        down_path = os.path.join(path, down_name)
        print('To download %s...  ' % down_name, end='')
        urllib.request.urlretrieve(down_link, down_path)
        print('Done!')


def download_string_files(path):
    make_dir(path)
    taxon_name_id_list = list(global_config.ID_MAP[name] for name in global_config.ID_MAP)
    taxon_id_set = set()
    for name_id in taxon_name_id_list:
        if not name_id[1] in taxon_id_set:
            taxon_id_set.add(name_id[1])
            down_path = os.path.join(path, name_id[0]+' '+name_id[1])
            download_string_file(down_path, name_id[1])
            time.sleep(60)
    print('Finish download string files')


def download_string_file(path, taxon_id):
    download_url = 'https://string-db.org/cgi/download.pl'+'?species_text=%s' % taxon_id
    reaponse = requests.get(download_url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    links = soup.find_all('div', {'class': 'download_table_data_row'})
    links_selected = list((links[3], links[6], links[7]))  # protein.actions.v11.0.txt.gz,protein.info.v11.0.txt.gz,protein.sequences.v11.0.fa.gz
    for index, link in enumerate(links_selected):
        down_link = link.find('a').attrs['href']
        down_name = global_config.STRING_FILES[index]+'.gz'
        down_path = os.path.join(path, down_name)
        down_env = internet_environment()
        down_file = requests.get(down_link, headers=down_env['headers'])
        with open(down_path, 'wb') as file:
            file.write(down_file.content)

    print(len(links), ' ', taxon_id)


def download_uniport_files(path):
    pass


'''
def download_essens_file(director, url):  # 在deg网站下载关键基因的所有数据，数据结构为基因名字和生物名字的数据对
    make_dir(director)
    reaponse = requests.get(url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    table = soup.find('table')
    links = table.find_all('a')
    for link in links:
        down_link = 'http://origin.tubic.org'+link.attrs['href']
        down_name = link.attrs['href'].split('/')[-1]
        down_dir = director + down_name
        print('To download %s....' % down_name, end=' ')
        urllib.request.urlretrieve(down_link, down_dir)
        print('Done!')
'''


def download_pnet_info(director, organ, url):  # 在string网站下载对应生物的蛋白质数据，包括蛋白质网络数据和蛋白质描述数据
    make_dir(director)
    temp_url = url+'?species_text=%s' % organ
    reaponse = requests.get(temp_url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    divs = soup.find_all('div', {'class': 'download_table_data_row'})
    links = []
    for div in divs:
        links.append(div.find('a').attrs['href'])

    uagent = UserAgent()
    chrome_uagent = uagent.data_browsers['chrome']
    headers = {'User-Agent': random.choice(chrome_uagent)}

    interact_link = links[3]
    interact_dir = director+'protein.actions.txt.gz'
    interact_downfile = requests.get(interact_link, headers=headers)
    with open(interact_dir, 'wb') as file:
        file.write(interact_downfile.content)
    protinfo_link = links[6]
    protinfo_dir = director+'protein.info.txt.gz'
    protinfo_downfile = requests.get(protinfo_link, headers=headers)
    with open(protinfo_dir, 'wb') as file:
        file.write(protinfo_downfile.content)


def download_protein_info(director, protein, url):
    pass


def get_related_organism(director):  # 在关键基因的所有数据中提取相关的生物
    filename_list = os.listdir(director)
    organ_set = set()
    for filename in filename_list:
        sign = filename.split('-')[1]
        temp_zip = zipfile.ZipFile(director+filename, 'r')
        annotation_file = temp_zip.open('degannotation-%s.dat' % sign)
        for line in annotation_file.readlines()[1:]:
            data = str(line).split(r'\t')
            organ_name = data[7]
            organ_set.add(organ_name)
    result = [organ for organ in list(organ_set) if (organ[:3] != 'NC_' and organ != '-')]  # 去除记录错误的部分
    return result


def get_origin_data(director):
    '''
    在网上爬取的原始数据
    Essens_info是从http://www.essentialgene.org/ 下载的7个文件,存储于Data/Origin_data/Essens_info/
    Protein_file是查看Essense_info中出现过的生物，从https://string-db.org/ 爬取文件，按照生物类别建立文件夹，存储于Data/Origin_data/Protein_file/
    '''
    # make_dir(director)
    essens_file_dir = director+'Essens_file/'
    # download_essens_file(essens_file_dir, r'http://origin.tubic.org/deg/public/index.php/download')
    organ_degname_list = get_related_organism(essens_file_dir)
    '''
    organ_degname_list = ['Bacillus subtilis 168', 'Agrobacterium fabrum str. C58 chromosome circular', 'Staphylococcus aureus NCTC 8325', 'Salmonella enterica serovar Typhimurium SL1344',
                  'Agrobacterium fabrum str. C58 chromosome linear', 'Bacteroides thetaiotaomicron VPI-5482', 'Mycobacterium tuberculosis H37Rv II', 'Caenorhabditis elegans', 'Salmonella typhimurium LT2',
                  'Streptococcus agalactiae A909', 'Drosophila melanogaster', 'Rhodopseudomonas palustris CGA009', 'Salmonella enterica subsp. enterica serovar Typhimurium str. 14028S',
                  'Mycobacterium tuberculosis H37Rv III', 'Bacillus thuringiensis BMB171', 'Escherichia coli MG1655 II', 'Haemophilus influenzae Rd KW20', 'Acinetobacter baumannii ATCC 17978',
                  'Streptococcus pyogenes NZ131', 'Streptococcus sanguinis', 'Caulobacter crescentus', 'Mycobacterium tuberculosis H37Rv', 'Mus musculus', 'Bacillus thuringiensis BMB171 plasmid pBMB171',
                  'Salmonella enterica serovar Typhi', 'Mycoplasma genitalium G37', 'Saccharomyces cerevisiae', 'Staphylococcus aureus N315', 'Campylobacter jejuni subsp. jejuni 81-176', 'Burkholderia thailandensis E264',
                  'Aspergillus fumigatus', 'Escherichia coli ST131 strain EC958', 'Schizosaccharomyces pombe 972h-', 'Porphyromonas gingivalis ATCC 33277', 'Synechococcus elongatus PCC 7942', 'Acinetobacter baylyi ADP1',
                  'Pseudomonas aeruginosa PAO1', 'Arabidopsis thaliana', 'Salmonella enterica serovar Typhi Ty2', 'Escherichia coli MG1655 I', 'Helicobacter pylori 26695', 'Synechococcus elongatus PCC 7942 plasmid 1',
                  'Brevundimonas subvibrioides ATCC 15264', 'Bacteroides fragilis 638R', 'Streptococcus pyogenes MGAS5448', 'Shewanella oneidensis MR-1', 'Vibrio cholerae N16961', 'Danio rerio', 'Mycoplasma pulmonis UAB CTIP',
                  'Francisella novicida U112', 'Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819', 'Streptococcus pneumoniae', 'Burkholderia pseudomallei K96243', 'Sphingomonas wittichii RW1', 'Homo sapiens',
                  'Pseudomonas aeruginosa UCBPP-PA14', 'Methanococcus maripaludis S2']
    '''
    # 由于名称的改动，需要做一定的人为调整
    organ_stringname_list = ['Bacillus subtilis 168', 'Agrobacterium fabrum str. C58', 'Staphylococcus aureus', 'Salmonella enterica serovar Typhimurium',
                             'Agrobacterium fabrum str. C58', 'Bacteroides thetaiotaomicron VPI-5482', 'Mycobacterium tuberculosis H37Rv', 'Caenorhabditis elegans', 'Salmonella typhimurium',
                             'None', 'Drosophila melanogaster', 'Rhodopseudomonas palustris CGA009', 'Salmonella enterica subsp. enterica serovar Typhimurium',
                             'Mycobacterium tuberculosis H37Rv', 'None', 'Escherichia coli MG1655', 'Haemophilus influenzae Rd KW20', 'Acinetobacter baumannii',
                             'Streptococcus pyogenes', 'None', 'None', 'Mycobacterium tuberculosis H37Rv', 'Mus musculus', 'None',
                             'Salmonella enterica serovar Typhimurium', 'Mycoplasma genitalium G37', 'Saccharomyces cerevisiae', 'Staphylococcus aureus', 'Campylobacter jejuni subsp. jejuni 81-176', 'None',
                             'Aspergillus fumigatus', 'None', 'Schizosaccharomyces pombe', 'Porphyromonas gingivalis ATCC 33277', 'Synechococcus elongatus PCC 7942', 'Acinetobacter baylyi ADP1',
                             'Pseudomonas aeruginosa', 'Arabidopsis thaliana', 'Salmonella enterica serovar Typhimurium', 'Escherichia coli MG1655', 'Helicobacter pylori 26695', 'Synechococcus elongatus PCC 7942',
                             'Brevundimonas subvibrioides ATCC 15264', 'None', 'Streptococcus pyogenes', 'Shewanella oneidensis', 'None', 'Danio rerio', 'Mycoplasma pulmonis UAB CTIP',
                             'None', 'Campylobacter jejuni subsp. jejuni NCTC 11168', 'None', 'Burkholderia pseudomallei K96243', 'Sphingomonas wittichii RW1', 'Homo sapiens',
                             'Pseudomonas aeruginosa', 'Methanococcus maripaludis S2']
    organ_set = set(organ_stringname_list)
    organ_set.remove('None')
    Pnet_info_dir = director+'Pnet_info/'
    make_dir(Pnet_info_dir)
    for organ in organ_set:
        Pnet_organ_dir = Pnet_info_dir+'%s/' % organ
        print('Begin download %s......' % organ, end='')
        download_pnet_info(Pnet_organ_dir, organ, r'https://string-db.org/cgi/download.pl')
        print('Finish')
        print('sleeping...')
        time.sleep(100)
    return organ_degname_list, organ_stringname_list


if __name__ == '__main__':
    main()