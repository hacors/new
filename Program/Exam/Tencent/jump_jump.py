import numpy as np

n = 6
data = [[1, 2, 3], [8, 9, 10], [5, 0, 5], [-9, -8, -10], [0, 1, 2], [5, 4, 6]]
'''
n = int(input())
data = []
for i in range(n):
    data.append(list(map(int, input().split())))
'''
dp_pos = np.zeros((n+1, 3), dtype=int)
dp_neg = np.zeros((n+1, 3), dtype=int)
for i in range(n):
    dp_pos[i+1][0] = max(dp_pos[i][0], dp_pos[i][1])+data[i][0]
    dp_pos[i+1][1] = max(dp_pos[i][0], dp_pos[i][1], dp_pos[i][2])+data[i][1]
    dp_pos[i+1][2] = max(dp_pos[i][1], dp_pos[i][2])+data[i][2]

    dp_neg[i+1][0] = max(dp_neg[i][0], dp_neg[i][1])-data[i][0]
    dp_neg[i+1][1] = max(dp_neg[i][0], dp_neg[i][1], dp_neg[i][2])-data[i][1]
    dp_neg[i+1][2] = max(dp_neg[i][1], dp_neg[i][2])-data[i][2]

    for j in range(3):
        if data[i][j] == 0:
            temp = dp_pos[i+1][j]
            dp_pos[i+1][j] = dp_neg[i+1][j]
            dp_neg[i+1][j] = temp
result = list(dp_neg[n])+list(dp_pos[n])
print(max(result))
