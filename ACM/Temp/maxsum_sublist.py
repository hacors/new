'''
relist = [2, -2, 7, 6, -8, -10, -5]
mylist = [-i for i in relist]
if mylist[0] >= 0:
    tempof1 = [mylist[0], mylist[0]+mylist[1]]
else:
    tempof1 = mylist[:2]
tempof2 = [0, mylist[0]+mylist[1]]
temp_max = max(tempof1)
result_max = 0
for temp in range(2, len(mylist)):
    if tempof1[temp-1] >= 0:
        tempof1.append(tempof1[temp-1]+mylist[temp])
    else:
        tempof1.append(mylist[temp])
    a = tempof2[temp-1]+mylist[temp]
    b = temp_max+mylist[temp]
    temp_max = max(temp_max, tempof1[temp])
    tempof2.append(max(a, b))
    result_max = max(result_max, tempof2[temp])
print(result_max*-1)
'''
import numpy as np
datas = [2, -2, 7, 6, -8, -10, -5]
k_num = 3

datas.insert(0, 0)
length = len(datas)
k_num = k_num+1
maxlist_here = np.zeros((k_num, length), dtype=int)
opti_solution = np.zeros((k_num, length), dtype=int)
for dim in range(1, k_num):
    for index in range(1, length):
        solution_append = maxlist_here[dim][index-1]+datas[index]
        solution_new = opti_solution[dim-1][index-1]+datas[index]
        maxlist_here[dim][index] = max(solution_append, solution_new)
        opti_solution[dim][index] = max(opti_solution[dim][index-1], maxlist_here[dim][index])
print(opti_solution[-1][-1])
