# EDIT THE FILE WITH YOUR SOLUTION
import re


def get_sentence(txt):
    article = str()
    for line in txt:
        line = line.replace('\n', ' ')
        article += line
    article = str.lower(article)
    article = re.sub('[,|:]', '', article)
    article = re.sub('[?|!]', '.', article)
    article = re.sub('[.]"', '".', article)
    all_sent = re.split('[.] ', article)
    useful_sent = all_sent[:-2]
    return useful_sent


def get_sirs(sent: str):
    sent = sent.replace('"', '')
    words = sent.split(' ')
    tempset = set()
    for index, word in enumerate(words):
        if word == 'sir':
            tempset.add(words[index+1])
        if word == 'sirs':
            indexofand = words.index('and')
            for i in range(indexofand-index-1):
                tempset.add(words[index+1+i])
            tempset.add(words[indexofand+1])
    return tempset


def get_imformation(sent):
    speak = re.search('"(.*)"', sent)
    if speak:
        imfo = speak.group(0)
        leftsent = sent.replace(imfo, '')
        speaker = get_sirs(leftsent).pop()
        number = sirs.index(speaker)
        imfo = imfo.replace('i ', 'sir %s ' % speaker)
        imfo = imfo.replace('knights', 'knight')
        imfo = imfo.replace('knaves', 'knave')
        imfo = imfo.replace('"', '')
        return [number, imfo]


def get_sentencelogic(sirs, imformation):
    # 将us统一换成members
    words = imformation.split(' ')
    if 'us' in words:
        numbers = range(len(sirs))
    else:
        related = list(get_sirs(imformation))
        numbers = list()
        for name in related:
            numbers.append(sirs.index(name))
    if 'knight' in words:
        mark = True
    else:
        mark = False
    if 'least' in words or 'or' in words:
        condi = 'least'
    elif 'most' in words:
        condi = 'most'
    elif 'exactly'in words:
        condi = 'exactly'
    else:
        condi = 'all'
    return [numbers, mark, condi]


def check(condition):
    tempcount = 0
    for num in condition[0]:
        if logiclist[num] == condition[1]:
            tempcount += 1
    if condition[2] == 'all':
        return tempcount == len(condition[0])
    if condition[2] == 'least':
        return tempcount >= 1
    if condition[2] == 'most':
        return tempcount <= 1
    if condition[2] == 'exactly':
        return tempcount == 1


def get_result(solution_num, solution_list, sirs):
    firstline = 'The sirs are:'
    for sir in sirs:
        firstline = firstline+' '+sir.capitalize()
    print(firstline)
    if solution_num == 0:
        print('There is no solution.')
    elif solution_num == 1:
        print('There is a unique solution:')
        for index, sir in enumerate(sirs):
            if solution_list[0][index]:
                who = 'Knight'
            else:
                who = 'Knave'
            print('Sir %s is a %s.' % (sir.capitalize(), who))
    else:
        print('There are %s solutions.' % solution_num)


direct = 'Practice/Bills/Knight_and_knaves/'  # 文件目录
txtname = input('Which text file do you want to use for the puzzle?')
txt = open(direct+txtname)

sents = get_sentence(txt)  # 切分句子

sirs = set()  # 得出所有人的名字
for sent in sents:
    sirs = sirs | get_sirs(sent)
sirs = list(sirs)
sirs.sort()

imfos = list()  # 得出有效信息
for sent in sents:
    imfo = get_imformation(sent)
    if(imfo):
        imfos.append(imfo)

for sent in imfos:  # 提取逻辑要求
    sent.append(get_sentencelogic(sirs, sent[1]))

logiclist = list()  # 逻辑计算
for i in range(len(sirs)):
    logiclist.append(False)

solutions = 0
solulist = list()
for test in range(pow(2, len(sirs))):
    temp = test
    for pos in range(len(sirs)):
        logiclist[-pos-1] = bool(temp % 2)
        temp = int(temp/2)
    result = True
    for condi in imfos:  # 对所有语句判断
        result = result & (not (check(condi[2]) ^ logiclist[condi[0]]))
    if result:  # 存储有效解
        solutions += 1
        solulist.append(logiclist.copy())
get_result(solutions, solulist, sirs)  # 输出结果
