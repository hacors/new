import os
import shutil
import requests
from bs4 import BeautifulSoup

ROOT = 'Graph/Protein/'


def make_dir(director):
    director = director[:-1] if director[-1] == '/' else director
    if os.path.exists(director):
        shutil.rmtree(director)
    os.mkdir(director)


def download_essens_file(director, url):
    reaponse = requests.get(url)
    soup = BeautifulSoup(reaponse.text, features='lxml')
    table = soup.find('table')
    links = table.find_all('a')
    for link in links:
        down_url = 'http://origin.tubic.org'+link.attrs['href']
        name = link.attrs['href'].split('/')[-1]
        '''
        req = requests.get(down_url)
        with open(director+name, 'wb') as file:
            file.write(req.content)
        '''


def get_origin_data(director):
    '''
    在网上爬取的原始数据
    Essens_info是从http://www.essentialgene.org/ 下载的7个文件,存储于Data/Origin_data/Essens_info/
    Protein_file是查看Essense_info中出现过的生物，从https://string-db.org/ 爬取文件，按照生物类别建立文件夹，存储于Data/Origin_data/Protein_file/
    '''
    make_dir(director)
    essens_info_dir = director+'Essens_file/'
    make_dir(essens_info_dir)
    download_essens_file(essens_info_dir, r'http://origin.tubic.org/deg/public/index.php/download')

    Protein_file_dir = director+'Protein_info/'


def main():
    data_dir = ROOT+'Data/'
    make_dir(data_dir)

    origin_data_dir = data_dir+'Origin_data/'
    get_origin_data(origin_data_dir)


if __name__ == '__main__':
    main()
