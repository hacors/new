#将一个字符串依次从左到右摘下来，执行放置左侧，放置右侧或者丢弃的操作，转换为另一个字符串
'''
n = int(input())
for i in range(n):
    mei = list(map(int, list(input())))
    bai = list(map(int, list(input())))

    maybe = pow(3, len(mei))
    result = []
    for num in range(maybe):
        choose = []
        left = []
        right = []
        temp_num = num
        for i in range(len(mei)):
            temp_choose = temp_num % 3
            temp_num = int(temp_num/3)
            if temp_choose == 0:
                choose.append('d')
            elif temp_choose == 1:
                left.insert(0, mei[i])
                choose.append('l')
            else:
                right.append(mei[i])
                choose.append('r')
        if left+right == bai:
            result.append(choose)
    print('{')
    for temp in result:
        print(temp[0], ' ', temp[1], ' ', temp[2], ' ')
    print('}')
'''

import sys


def dfs(mei, bai, temp, command):
    if len(temp) == len(bai):
        if temp == bai:
            for _ in mei:
                command.append('d')
            result.append(command)
    if mei and len(temp) < len(bai):
        dfs(mei[1:], bai, temp, command+['d'])
        dfs(mei[1:], bai, [mei[0]]+temp, command+['l'])
        dfs(mei[1:], bai, temp+[mei[0]], command+['r'])


sample_num = int(sys.stdin.readline().strip())
for _ in range(sample_num):
    mei = list(sys.stdin.readline().strip())
    bai = list(sys.stdin.readline().strip())
    result = []
    temp = []
    command = []
    dfs(mei, bai, temp, command)
    print('{')
    for each in result:
        print(' '.join(each))
    print('}')
