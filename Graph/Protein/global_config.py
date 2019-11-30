'''
设置全局的配置文件，比如下载地址，运行环境等等
'''
import os
DATA_ROOT = os.path.join(os.getcwd(), 'Data_test')
DIR_LIST = {'DEG': os.path.join(DATA_ROOT, 'DEG'),
            'Id_map': os.path.join(DATA_ROOT, 'Id_map'),
            'STRING': os.path.join(DATA_ROOT, 'STRING'),
            'UNIPROT': os.path.join(DATA_ROOT, 'UNIPROT')
            }

ID_MAP = {'-': 0,
          'Acinetobacter baumannii ATCC 17978': '470',  # check
          'Acinetobacter baylyi ADP1': '62977',
          'Agrobacterium f': '176299',  # check
          'Arabidopsis thaliana': '3702',
          'Aspergillus fumigatus': '746128',
          'Bacillus subtilis 168': '224308',
          'Bacillus thuringiensis BMB171': '714359',  # no
          'Bacillus thuringiensis BMB171 plasmid pBMB171': '714359',  # no
          'Bacteroides fragilis 638R': '862962',  # check
          'Bacteroides thetaiotaomicron VPI-5482': '226186',
          'Brevundimonas subvibrioides ATCC 15264': '633149',
          'Burkholderia pseudomallei K96243': '272560',
          'Burkholderia thailandensis E264': '271848',  # no
          'Caenorhabditis elegans': '6239',
          'Campylobacter jejuni subsp. jejuni 81-176': '354242',
          'Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819': '192222',
          'Caulobacter crescentus': '565050',  # check
          'Danio rerio': '7955',
          'Drosophila melanogaster': '7227',
          'Escherichia coli MG1655 I': '511145',  # multi
          'Escherichia coli MG1655 II': '511145',  # multi
          'Escherichia coli ST131 strain EC958': '941332',  # check
          'Francisella novicida U112': '401614',  # check
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
          'Pseudomonas aeruginosa PAO1': '208964',  # change multi 287
          'Pseudomonas aeruginosa UCBPP-PA14': '208963',  # change multi 287
          'Rhodopseudomonas palustris CGA009': '258594',
          'Saccharomyces cerevisiae': '559292',  # change 4932
          'Salmonella enterica serovar Typhi': '209261',  # check
          'Salmonella enterica serovar Typhi Ty2': '209261',  # check
          'Salmonella enterica serovar Typhimurium SL1344': '216597',  # check
          'Salmonella enterica subsp. enterica serovar Typhimurium str. 14028S': '588858',  # check change 90371
          'Salmonella typhimurium LT2': '99287',  # check
          'Schizosaccharomyces pombe 972h-': '4896',
          'Shewanella oneidensis MR-1': '211586',
          'Sphingomonas wittichii RW1': '392499',
          'Staphylococcus aureus N315': '158879',  # change multi 1208
          'Staphylococcus aureus NCTC 8325': '93061',  # change multi 1208
          'Streptococcus agalactiae A909': '205921',  # check
          'Streptococcus pneumoniae': '170187',  # check
          'Streptococcus pyogenes MGAS5448': '293653',  # change multi 1314
          'Streptococcus pyogenes NZ131': '471876',  # change multi 1314
          'Streptococcus sanguinis': '388919',  # check
          'Synechococcus elongatus PCC 7942': '1140',  # multi
          'Synechococcus elongatus PCC 7942 plasmid 1': '1140',  # multi
          'Vibrio cholerae N16961': '243277'
          }
print(len(ID_MAP))
