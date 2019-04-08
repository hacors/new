num = 5
demand = [5, -4, 1, -3, 1]
cur = 0
result = 0
for i in range(len(demand)):
    if cur == 0:
        cur = demand[i]
    elif (cur > 0 and demand[i] < 0)or(cur < 0 and demand[i] > 0):
        result = result+min(abs(cur), abs(demand[i]))
        cur = cur+demand[i]
    else:
        cur = cur+demand[i]
print(result)
