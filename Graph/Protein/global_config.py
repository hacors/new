'''
设置全局的配置文件，比如下载地址，运行环境等等
'''
import os
DATA_ROOT = os.path.join('Data_test')
PATH_LIST = {'DEG': os.path.join(DATA_ROOT, 'DEG'),
             'STRING': os.path.join(DATA_ROOT, 'STRING'),
             'UNIPROT': os.path.join(DATA_ROOT, 'UNIPROT')
             }
DEG_BIO_CLASS = ['Bacteria', 'Archaea', 'Eukaryotes']
STRING_FILES = ['protein_actions', 'protein_info', 'protein_sequences']
ID_INFO = {'-': '0',
           'Acinetobacter baumannii ATCC 17978': '400667',  # 存在通用ID:470,name:Acinetobacter baumannii
           'Acinetobacter baylyi ADP1': '62977',
           'Agrobacterium f': '176299',  # name:Agrobacterium fabrum str. C58 chromosome circular
           'Arabidopsis thaliana': '3702',
           'Aspergillus fumigatus': '746128',
           'Bacillus subtilis 168': '224308',
           'Bacillus thuringiensis BMB171': '714359',  # multi 不存在对应关系
           'Bacillus thuringiensis BMB171 plasmid pBMB171': '714359',  # multi 不存在对应关系
           'Bacteroides fragilis 638R': '862962',  # 不存在对应关系，但是存在其他变种（ID:272559,name:Bacteroides fragilis NCTC 9343;ID:457424,name:Bacteroides fragilis 3_1_12)
           'Bacteroides thetaiotaomicron VPI-5482': '226186',
           'Brevundimonas subvibrioides ATCC 15264': '633149',
           'Burkholderia pseudomallei K96243': '272560',
           'Burkholderia thailandensis E264': '271848',  # 不存在的对应关系
           'Caenorhabditis elegans': '6239',
           'Campylobacter jejuni subsp. jejuni 81-176': '354242',
           'Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819': '192222',
           'Caulobacter crescentus': '565050',  # name:Caulobacter crescentus NA1000,不存在对应关系，但是存在其他变种（ID:190650,name:Caulobacter crescentus CB15;ID:1292034,name:Caulobacter crescentus OR37)
           'Danio rerio': '7955',
           'Drosophila melanogaster': '7227',
           'Escherichia coli MG1655 I': '511145',  # multi,name:Escherichia coli str. K-12 substr. MG1655
           'Escherichia coli MG1655 II': '511145',  # multi,name:Escherichia coli str. K-12 substr. MG1655
           'Escherichia coli ST131 strain EC958': '941332',  # 不存在对关系，但是存在多个其他变种
           'Francisella novicida U112': '401614',  # 不存在的对应关系，但是存在其他变种(ID:676032,name:Francisella cf. tularensis subsp. novicida 3523)
           'Haemophilus influenzae Rd KW20': '71421',
           'Helicobacter pylori 26695': '85962',
           'Homo sapiens': '9606',  # important
           'Methanococcus maripaludis S2': '267377',
           'Mus musculus': '10090',  # important
           'Mycobacterium tuberculosis H37Rv': '83332',  # multi
           'Mycobacterium tuberculosis H37Rv II': '83332',  # multi
           'Mycobacterium tuberculosis H37Rv III': '83332',  # multi
           'Mycoplasma genitalium G37': '243273',
           'Mycoplasma pulmonis UAB CTIP': '272635',
           'Porphyromonas gingivalis ATCC 33277': '431947',
           'Pseudomonas aeruginosa PAO1': '208964',  # multi,存在通用ID:287,name:Pseudomonas aeruginosa
           'Pseudomonas aeruginosa UCBPP-PA14': '208963',  # multi,存在通用ID:287,name:Pseudomonas aeruginosa
           'Rhodopseudomonas palustris CGA009': '258594',
           'Saccharomyces cerevisiae': '559292',  # name:Saccharomyces cerevisiae S288C,存在通用ID:4932
           'Salmonella enterica serovar Typhi': '209261',  # 不存在对应关系，存在其他变种(220341,90371)
           'Salmonella enterica serovar Typhi Ty2': '209261',  # 不存在对应关系，存在其他变种(220341,90371)
           'Salmonella enterica serovar Typhimurium SL1344': '216597',  # 不存在对应关系，存在其他变种(220341,90371)
           'Salmonella enterica subsp. enterica serovar Typhimurium str. 14028S': '588858',  # 不存在对应关系，存在其他变种(220341,90371)
           'Salmonella typhimurium LT2': '99287',  # 不存在对应关系
           'Schizosaccharomyces pombe 972h-': '4896',
           'Shewanella oneidensis MR-1': '211586',
           'Sphingomonas wittichii RW1': '392499',
           'Staphylococcus aureus N315': '158879',  # multi,存在通用ID:1280,name:Staphylococcus aureus
           'Staphylococcus aureus NCTC 8325': '93061',  # multi,存在通用ID:1280,name:Staphylococcus aureus
           'Streptococcus agalactiae A909': '205921',  # 不存在对应关系，存在其他变种(1154860,211110)
           'Streptococcus pneumoniae': '170187',  # 存在一对多的关系，无法分割，舍弃(Streptococcus pneumoniae R6,171010,Streptococcus pneumoniae TIGR4,171018)
           'Streptococcus pyogenes MGAS5448': '293653',  # multi,存在通用ID:1314,name:Streptococcus pyogenes
           'Streptococcus pyogenes NZ131': '471876',  # multi,存在通用ID:1314,name:Streptococcus pyogenes
           'Streptococcus sanguinis': '388919',  # name:Streptococcus sanguinis SK36
           'Synechococcus elongatus PCC 7942': '1140',  # multi
           'Synechococcus elongatus PCC 7942 plasmid 1': '1140',  # multi
           'Vibrio cholerae N16961': '243277'
           }

ID_MAP = {'Acinetobacter baumannii ATCC 17978': ['Acinetobacter_baumannii', '470'],
          'Acinetobacter baylyi ADP1': ['Acinetobacter_baylyi', '62977'],
          'Agrobacterium f': ['Agrobacterium', '176299'],
          'Arabidopsis thaliana': ['Arabidopsis_thaliana', '3702'],
          'Aspergillus fumigatus': ['Aspergillus_fumigatus', '746128'],
          'Bacillus subtilis 168': ['Bacillus_subtilis', '224308'],
          'Bacteroides thetaiotaomicron VPI-5482': ['Bacteroides_thetaiotaomicron', '226186'],
          'Brevundimonas subvibrioides ATCC 15264': ['Brevundimonas_subvibrioides', '633149'],
          'Burkholderia pseudomallei K96243': ['Burkholderia_pseudomallei', '272560'],
          'Caenorhabditis elegans': ['Caenorhabditis_elegans', '6239'],
          'Campylobacter jejuni subsp. jejuni 81-176': ['Campylobacter_jejuni_81_176', '354242'],
          'Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819': ['Campylobacter_jejuni_NCTC', '192222'],
          'Danio rerio': ['Danio_rerio', '7955'],
          'Drosophila melanogaster': ['Drosophila_melanogaster', '7227'],
          'Escherichia coli MG1655 I': ['Escherichia_coli', '511145'],
          'Escherichia coli MG1655 II': ['Escherichia_coli', '511145'],
          'Haemophilus influenzae Rd KW20': ['Haemophilus_influenzae', '71421'],
          'Helicobacter pylori 26695': ['Helicobacter_pylori', '85962'],
          'Homo sapiens': ['Homo_sapiens', '9606'],
          'Methanococcus maripaludis S2': ['Methanococcus_maripaludis', '267377'],
          'Mus musculus': ['Mus_musculus', '10090'],
          'Mycobacterium tuberculosis H37Rv': ['Mycobacterium_tuberculosis', '83332'],
          'Mycobacterium tuberculosis H37Rv II': ['Mycobacterium_tuberculosis', '83332'],
          'Mycobacterium tuberculosis H37Rv III': ['Mycobacterium_tuberculosis', '83332'],
          'Mycoplasma genitalium G37': ['Mycoplasma_genitalium', '243273'],
          'Mycoplasma pulmonis UAB CTIP': ['Mycoplasma_pulmonis', '272635'],
          'Porphyromonas gingivalis ATCC 33277': ['Porphyromonas_gingivalis', '431947'],
          'Pseudomonas aeruginosa PAO1': ['Pseudomonas_aeruginosa', '287'],
          'Pseudomonas aeruginosa UCBPP-PA14': ['Pseudomonas_aeruginosa', '287'],
          'Rhodopseudomonas palustris CGA009': ['Rhodopseudomonas_palustris', '258594'],
          'Saccharomyces cerevisiae': ['Saccharomyces_cerevisiae', '4932'],
          'Schizosaccharomyces pombe 972h-': ['Schizosaccharomyces_pombe', '4896'],
          'Shewanella oneidensis MR-1': ['Shewanella_oneidensis', '211586'],
          'Sphingomonas wittichii RW1': ['Sphingomonas_wittichii', '392499'],
          'Staphylococcus aureus N315': ['Staphylococcus_aureus', '1280'],
          'Staphylococcus aureus NCTC 8325': ['Staphylococcus_aureus', '1280'],
          'Streptococcus pyogenes MGAS5448': ['Streptococcus_pyogenes', '1314'],
          'Streptococcus pyogenes NZ131': ['Streptococcus_pyogenes', '1314'],
          'Streptococcus sanguinis': ['Streptococcus_sanguinis', '388919'],
          'Synechococcus elongatus PCC 7942': ['Synechococcus_elongatus', '1140'],
          'Synechococcus elongatus PCC 7942 plasmid 1': ['Synechococcus_elongatus', '1140'],
          'Vibrio cholerae N16961': ['Vibrio_cholerae', '243277']
          }
if __name__ == '__main__':
    print(len(ID_MAP))
