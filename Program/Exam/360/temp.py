lenght = 10
step_num = 3
step_list = [6, 2, 5]

step_list = step_list.reverse()
pos = 0
left = 0
right = 0
for step in step_list:
    temp_left = pos-left
    temp_right = right-pos
    the_max = max(abs(temp_left), abs(temp_right))
    if the_max == abs(temp_right):
        pos = pos+step
        if pos > right:
            right = pos
    else:
        pos = pos-step
        if pos < left:
            left = pos
print(left, right)
