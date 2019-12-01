'''
在网上爬取的原始数据
从http://www.essentialgene.org/ 下载所有的关键蛋白信息
从https://www.ncbi.nlm.nih.gov/ 查找不同标准的生物的ID对应关系，存储于单独的文件夹
从https://string-db.org/ 下载所有相关生物的蛋白质相互作用信息，按照生物ID分类存储
从https://www.uniprot.org/ 查找所有相关的蛋白质注释信息，按照生物ID分类存储
'''
import gzip
import os
import random
import shutil
import time
import urllib
import zipfile

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

import global_config
from selenium import webdriver


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
    # download_string_files(path=path_list['STRING'])
    download_uniport_files(string_path=path_list['STRING'], path=path_list['UNIPROT'])
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
        print('To download deg %s ...... ' % down_path, end='')
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


def download_string_file(path, taxon_id):
    make_dir(path)
    download_url = 'https://string-db.org/cgi/download.pl'+'?species_text=%s' % taxon_id
    reaponse = requests.get(download_url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    links = soup.find_all('div', {'class': 'download_table_data_row'})
    links_selected = list((links[3], links[6], links[7]))  # protein.actions.v11.0.txt.gz,protein.info.v11.0.txt.gz,protein.sequences.v11.0.fa.gz
    for index, link in enumerate(links_selected):
        down_link = link.find('a').attrs['href']
        down_name = global_config.STRING_FILES[index]+'.txt.gz'
        down_path = os.path.join(path, down_name)
        down_env = internet_environment()
        print('To download string %s ...... ' % down_path, end='')
        down_file = requests.get(down_link, headers=down_env['headers'])
        with open(down_path, 'wb') as file:
            file.write(down_file.content)
            time.sleep(60)
        print('Done!')


def download_uniport_files(string_path, path):
    make_dir(path)
    string_organ_list = os.listdir(string_path)
    for string_organ in string_organ_list:
        gz_file_path = os.path.join(string_path, string_organ, global_config.STRING_FILES[1]+'.txt.gz')
        protein_id_list = list()
        with gzip.open(gz_file_path, 'r') as file:
            next(file)
            for row in file:
                row_split = str(row, encoding='utf-8').split('\t')
                protein_id_list.append(row_split[0])
        down_path = os.path.join(path, string_organ)
        download_uniport_file(down_path, protein_id_list)


def download_uniport_file(down_path, protein_id_list):
    make_dir(down_path)
    # 存储所有需要的蛋白质id
    temp_file_path = os.path.join(down_path, 'protein_id_list.txt')
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        for protein_id in protein_id_list:
            file.write(protein_id+'\n')
    # 打开并设置浏览器
    options = webdriver.ChromeOptions()
    prefs = {'profile.default_content_settings.popups': 0, 'download.default_directory': os.path.abspath(down_path)}
    options.add_experimental_option('prefs', prefs)
    options.add_argument('--headless')
    driver = webdriver.Chrome(chrome_options=options)
    # 获取查询的下载链接
    driver.get('https://www.uniprot.org/uploadlists/')
    select = webdriver.support.select.Select(driver.find_element_by_id('from-database'))
    select.select_by_value('STRING_ID')
    driver.find_element_by_id('uploadfile').send_keys(os.path.abspath(temp_file_path))
    driver.find_element_by_id('upload-submit').click()
    uniprot_list = driver.find_element_by_id('query').get_attribute('value')
    uniprot_id = uniprot_list.replace('yourlist:', '')
    final_url = 'https://www.uniprot.org/uniprot/?query=yourlist:%s&sort=yourlist:%s&columns=yourlist(%s)' % (uniprot_id, uniprot_id, uniprot_id) + \
        ',id,entry%20name,protein%20names,genes,organism,length,features,go-id'  # 初步使用的特征
    # 开始下载
    driver.get(final_url)
    driver.find_element_by_id('download-button').click()
    select = webdriver.support.select.Select(driver.find_element_by_id('format'))
    select.select_by_value('tab')
    driver.find_element_by_id('menu-go').click()  # 下载文件
    # 改名
    while not os.path.exists():
        os.rename(os.path.join(down_abs_path, 'uniprot-yourlist_%s.tab.gz' % uniprot_id), os.path.join(down_abs_path, '123.gz'))
    driver.quit()


if __name__ == '__main__':
    main()
