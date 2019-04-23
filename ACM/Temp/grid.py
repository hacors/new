'''
n = int(input())
operate = list()
for i in range(n):
    operate.append(list(map(int, input().split(' '))))
'''
operate = [[1, 2, 3, 2], [2, 5, 2, 3], [1, 4, 3, 4], [3, 2, 4, 2], [2, 2, 1, 2]]
the_col, the_row = {}, {}
for element in operate:
    if element[0] == element[2]:
        if element[0] in the_col:
            the_col[element[0]].append([element[1], element[3]])
        else:
            the_col[element[0]] = [[element[1], element[3]]]
    else:
        if element[1] in the_row:
            the_row[element[1]].append([element[0], element[2]])
        else:
            the_row[element[1]] = [[element[0], element[2]]]

print(operate)
