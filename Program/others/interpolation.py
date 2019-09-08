def interpolation(position, num):  # 牛顿迭代法
    while abs(position*position-num) > 1e-15:
        position = position-(position*position-num)/(2*position)
        print(position)
    return position


result = interpolation(1000000, 2)
pass
