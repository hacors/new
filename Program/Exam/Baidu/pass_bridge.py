num, storage = 4, 2
weight_list = [1, 1, 1, 1]
time_list = [2, 1, 2, 2]
'''
num, storage = list(map(int, input().split()))
weight_list = list(map(int, input().split()))
time_list = list(map(int, input().split()))
'''
weight_list.insert(0, storage)
time_list.insert(0, 1)

on_bridge_index = [0]
time = 0
can_add = 0
head_car = 1
smallest = 1
while on_bridge_index:
    temp = time_list[on_bridge_index]
    smallest = min(time_list[on_bridge_index])
    time += smallest
    remove_list = []
    for index in on_bridge_index:
        time_list[index] -= smallest
        if time_list[index] == 0:
            can_add += weight_list[index]
            remove_list.append(index)
    for removed in remove_list:
        on_bridge_index.remove(removed)
    while head_car < len(weight_list) and can_add >= weight_list[head_car]:
        on_bridge_index.append(head_car)
        can_add -= weight_list[head_car]
        head_car += 1
print(time-1)
