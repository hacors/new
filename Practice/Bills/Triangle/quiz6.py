# Randomly generates a grid with 0s and 1s, whose dimension is controlled by user input,
# as well as the density of 1s in the grid, and determines the size of the largest
# isosceles triangle, consisting of nothing but 1s and whose base can be either
# vertical or horizontal, pointing either left or right or up or down.
#
# Written by *** and Eric Martin for COMP9021

from random import seed, randint
import copy
import sys


def display_grid():
    for i in range(len(grid)):
        print('   ', ' '.join(str(int(grid[i][j] != 0)) for j in range(len(grid))))


def grid_change(the_grid):  # 将数据的格式处理一下
    zeros = [0]*(len(the_grid)+2)
    result = list([zeros])
    for the_list in the_grid:
        temp = [0] + list((int(the_list[index] != 0)) for index in range(len(the_list)))+[0]
        result.append(temp)
    result.append(zeros)
    return result


def grid_rotate(the_grid):  # 旋转数据
    length = len(the_grid)
    for i in range(length):
        for j in range(i+1, length):
            temp = the_grid[i][j]
            the_grid[i][j] = the_grid[j][i]
            the_grid[j][i] = temp
        the_grid[i].reverse()
    return the_grid


def has_one(the_grid):  # 判断数据中是否还有1
    for the_list in the_grid:
        if 1 in the_list:
            return True
    return False


def simple_check(the_grid, size):  # 检查是否有简单三角存在
    length = len(the_grid)
    for i in range(1, length-1):
        for j in range(1, length-1):
            if not (the_grid[i][j] and the_grid[i+1][j] and the_grid[i+1][j-1] and the_grid[i+1][j+1]):
                the_grid[i][j] = 0
    return the_grid, size+1


def size_of_largest_isosceles_triangle():
    my_grid = grid_change(grid)
    maxnum = 0
    for trave in range(4):  # 判断四个方向
        my_grid = grid_rotate(my_grid)
        grid_temp = copy.deepcopy(my_grid)
        tempnum = 0
        while(has_one(grid_temp)):
            grid_temp, tempnum = simple_check(grid_temp, tempnum)
        maxnum = max(tempnum, maxnum)
    return maxnum
    # REPLACE pass WITH YOUR CODE

# POSSIBLY DEFINE OTHER FUNCTIONS


try:
    arg_for_seed, density = (abs(int(x)) for x in input('Enter two integers: ').split())
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()
seed(arg_for_seed)
grid = [[randint(0, density) for _ in range(10)] for _ in range(10)]
print('Here is the grid that has been generated:')
display_grid()
print('The largest isosceles triangle has a size of',
      size_of_largest_isosceles_triangle()
      )
