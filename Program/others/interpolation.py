def interpolation(position, num):
    while abs(position*position-num) > 1e-15:
        position = position-(position*position-num)/(2*position)
        print(position)
    return position


result = interpolation(1000000, 2)
pass
