import random
import math
import sys
import heapq
random.seed(666)


class MedianFinder():
    # 数据流中的中位数
    def __init__(self):
        self.low_heap = []
        self.high_heap = []
        self.nums = 0

    def addNum(self, num):  # 注意简单的做法是先保证形状的符合，再保证内容的符合
        if self.nums % 2:
            heapq.heappush(self.high_heap, num)
        else:
            heapq.heappush(self.low_heap, -num)
        self.nums += 1
        self.adjust()

    def adjust(self):
        if self.nums > 1:
            while -self.low_heap[0] > self.high_heap[0]:
                temp = -heapq.heappop(self.low_heap)
                temp = heapq.heappushpop(self.high_heap, temp)
                heapq.heappush(self.low_heap, -temp)

    def findMedian(self):
        if self.nums % 2:
            return -self.low_heap[0]
        else:
            return (-self.low_heap[0]+self.high_heap[0])/2


class UpsetArray():
    def __init__(self, nums: list):
        self.origin_nums = nums

    def reset(self):
        return self.origin_nums

    def shuffle(self):
        temp_nums = self.origin_nums.copy()
        for index in range(len(temp_nums)):
            choose_index = random.randint(index, len(temp_nums)-1)
            temp_nums[index], temp_nums[choose_index] = temp_nums[choose_index], temp_nums[index]
        return temp_nums


class RandomizedCollection:
    def __init__(self):


    def insert(self, val: int) -> bool:


    def remove(self, val: int) -> bool:


    def getRandom(self) -> int:



if __name__ == '__main__':
    solu = MedianFinder()
    '''
    ["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
    [[],[1],[2],[],[3],[]]
    '''
    pass
