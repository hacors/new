mylist = list(map(int, input()))
# mylist = [1, 0, 1, 0, 1]
index_zero = -1
for index in range(len(mylist)):
    if mylist[index] == 0:
        index_zero = index
        break
if index_zero == -1:
    print(len(mylist))
else:
    newlist = [0]+mylist[index_zero+1:]+mylist[0:index_zero]
    temp = [0]*len(newlist)
    result = 0
    for i in range(1, len(newlist)):
        if newlist[i] == 1:
            temp[i] = temp[i-1]+1
            result = max(result, temp[i])
    print(result)
