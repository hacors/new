'''
n = 3
m = 10
number = [2, 5, 3]
price = [2, 1, 3]
'''
n, m = list(map(int, input().split()))
number = list(map(int, input().split()))
price = list(map(int, input().split()))

combin = []
for i in range(len(number)):
    combin.append([number[i], price[i]])
combin = sorted(combin, key=lambda x: x[0])
should_minus = 0
left_money = m
cake_num = 0
index = 0
while left_money > 0:
    cake_num += 1
    while combin[index][0] < cake_num:
        should_minus += combin[index][1]
        index += 1
    left_money -= should_minus
print(cake_num-1)
