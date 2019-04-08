if mylist[0]  0:
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
