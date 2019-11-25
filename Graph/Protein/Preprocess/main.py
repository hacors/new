import os
import shutil
import urllib
import zipfile
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter

ROOT = 'Graph/Protein/'


def make_dir(director):
    director = director[:-1] if director[-1] == '/' else director
    if os.path.exists(director):
        shutil.rmtree(director)
    os.mkdir(director)


def download_essens_file(director, url):  # 在网上下载关键基因的所有数据
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


def download_pnet_info(director, organ, url):
    make_dir(director)
    temp_url = url+'?species_text=%s' % organ
    reaponse = requests.get(temp_url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    divs = soup.find_all('div', {'class': 'download_table_data_row'})
    links = []
    for div in divs:
        links.append(div.find('a').attrs['href'])

    headers = {
        'Accept-Language': 'zh-CN',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19'}
    interact_link = links[3]
    interact_dir = director+'protein.actions.txt.gz'
    urllib.request.Request(interact_link, None, headers)
    urllib.request.urlretrieve(interact_link, interact_dir)

    protinfo_link = links[6]
    protinfo_dir = director+'protein.info.txt.gz'
    urllib.request.urlretrieve(protinfo_link, protinfo_dir)
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
    organ_list = get_related_organism(essens_file_dir)
    '''
    organ_list = ['Bacillus subtilis 168', 'Agrobacterium fabrum str. C58 chromosome circular', 'Staphylococcus aureus NCTC 8325', 'Salmonella enterica serovar Typhimurium SL1344',
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
    corrected_organs = ['Bacillus subtilis 168', 'Agrobacterium fabrum str. C58', 'Staphylococcus aureus', 'Salmonella enterica serovar Typhimurium',
                        'Agrobacterium fabrum str. C58', 'Bacteroides thetaiotaomicron VPI-5482', 'Mycobacterium tuberculosis H37Rv', 'Caenorhabditis elegans', 'Salmonella typhimurium',
                        'None', 'Drosophila melanogaster', 'Rhodopseudomonas palustris CGA009', 'Salmonella enterica subsp. enterica serovar Typhimurium',
                        'Mycobacterium tuberculosis H37Rv', 'None', 'Escherichia coli MG1655', 'Haemophilus influenzae Rd KW20', 'Acinetobacter baumannii',
                        'Streptococcus pyogenes', 'None', 'None', 'Mycobacterium tuberculosis H37Rv', 'Mus musculus', 'None',
                        'Salmonella enterica serovar Typhimurium', 'Mycoplasma genitalium G37', 'Saccharomyces cerevisiae', 'Staphylococcus aureus', 'Campylobacter jejuni subsp. jejuni 81-176', 'Burkholderia thailandensis E264',
                        'Aspergillus fumigatus', 'None', 'Schizosaccharomyces pombe', 'Porphyromonas gingivalis ATCC 33277', 'Synechococcus elongatoneus PCC 7942', 'Acinetobacter baylyi ADP1',
                        'Pseudomonas aeruginosa', 'Arabidopsis thaliana', 'Salmonella enterica serovar Typhimurium', 'Escherichia coli MG1655', 'Helicobacter pylori 26695', 'Synechococcus elongatus PCC 7942',
                        'Brevundimonas subvibrioides ATCC 15264', 'None', 'Streptococcus pyogenes', 'Shewanella oneidensis', 'None', 'Danio rerio', 'Mycoplasma pulmonis UAB CTIP',
                        'None', 'Campylobacter jejuni subsp. jejuni NCTC 11168', 'None', 'Burkholderia pseudomallei K96243', 'Sphingomonas wittichii RW1', 'Homo sapiens',
                        'Pseudomonas aeruginosa', 'Methanococcus maripaludis S2']
    organ_list = list(set(corrected_organs))
    organ_list.remove('None')
    Pnet_info_dir = director+'Pnet_info/'
    make_dir(Pnet_info_dir)
    for organ in organ_list:
        Pnet_organ_dir = Pnet_info_dir+'%s/' % organ
        download_pnet_info(Pnet_organ_dir, organ, r'https://string-db.org/cgi/download.pl')


def main():
    data_dir = ROOT+'Data/'
    # make_dir(data_dir)

    origin_data_dir = data_dir+'Origin_data/'
    get_origin_data(origin_data_dir)


if __name__ == '__main__':
    main()
