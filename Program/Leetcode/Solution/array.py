import random
import math
import sys
random.seed(666)


class Solution():
    def findMedianSortedArrays(self, nums1: list, nums2: list):
        '''
        解法一：关键在于找到两个列表的各自切分点，这个切分点满足中位数，而这个切分点的寻找过程需要使用到二分法
        解法二：需要找到第k个元素，可以利用数据的有序性不断削减问题规模
        解法三：各自找中位数，去除不符合的部分，依次迭代(这个只适用于两个等长序列)
        '''
        # 解法一
        if len(nums1) < len(nums2):
            nums1, nums2 = nums2, nums1
        if len(nums2) == 0:
            return nums1[len(nums1)//2] if len(nums1) % 2 else (nums1[len(nums1)//2]+nums1[len(nums1)//2-1])/2
        left_bound = (len(nums1)-len(nums2))//2
        right_bound = (len(nums1)+len(nums2))//2
        while left_bound <= right_bound:
            mid_1 = (left_bound+right_bound)//2
            mid_2 = (len(nums1)+len(nums2))//2-mid_1
            if mid_1 > 0 and mid_2 < len(nums2) and nums1[mid_1-1] > nums2[mid_2]:
                right_bound = mid_1-1
            elif mid_1 < len(nums1) and mid_2 > 0 and nums1[mid_1] < nums2[mid_2-1]:
                left_bound = mid_1+1
            else:
                if mid_1 < len(nums1) and mid_2 < len(nums2):
                    right_min = min(nums1[mid_1], nums2[mid_2])
                else:
                    right_min = nums1[mid_1] if mid_1 < len(nums1) else nums2[mid_2]
                if (len(nums1)+len(nums2)) % 2:
                    return right_min
                if mid_1 > 0 and mid_2 > 0:
                    left_max = max(nums1[mid_1-1], nums2[mid_2-1])
                else:
                    left_max = nums1[mid_1-1] if mid_1 > 0 else nums2[mid_2-1]
                return (left_max+right_min)/2

    def sortArray(self, nums: list):
        # 栈插入排序，超时
        '''
        if not len(nums):
            return nums
        temp = nums.pop()
        sortarray = []
        while len(nums) or (len(sortarray) and sortarray[-1] > temp):  # 用栈实现插入排序的移动操作，关键在于最后一个元素需要保证其大于等于排序数组的栈顶
            if len(sortarray) == 0 or temp >= sortarray[-1]:
                sortarray.append(temp)
                temp = nums.pop()
            else:
                nums.append(sortarray.pop())
        sortarray.append(temp)
        return sortarray
        '''
        '''        
        # 快排实现一：从两端往中间分割的方法
        # 既然需要两端交替，那么可以安排每一轮两端都看一下，使用while循环嵌套while循环
        def temp_sort(left, right):  # 一定要清楚left和right代表的意义
            if left == right:
                return
            refer = nums[left]
            left_pos, right_pos = left, right-1
            while left_pos < right_pos:  # 使用三个循环实现左右左右左右的依次操作
                while left_pos < right_pos and nums[right_pos] >= refer:
                    right_pos -= 1
                nums[left_pos] = nums[right_pos]
                while left_pos < right_pos and nums[left_pos] < refer:
                    left_pos += 1
                nums[right_pos] = nums[left_pos]
            midindex = left_pos
            nums[midindex] = refer  # 最后需要放回取出的refer元素
            temp_sort(left, midindex)
            temp_sort(midindex+1, right)  # 注意refer在的位置是已经确定的
        temp_sort(0, len(nums))
        return(nums)
        '''
        # 快排实现一：简化版
        def temp_sort(left, right):  # 此时选取的left和right为闭区间
            if left >= right:
                return
            refer = nums[left]  # 随机选择一个数
            left_pos, right_pos = left, right  # 定义遍历起点
            while left_pos < right_pos:  # 使用三个循环实现左右左右左右的依次操作
                while nums[right_pos] >= refer and left_pos < right_pos:
                    right_pos -= 1
                nums[left_pos] = nums[right_pos]
                while nums[left_pos] <= refer and left_pos < right_pos:
                    left_pos += 1
                nums[right_pos] = nums[left_pos]
            nums[left_pos] = refer
            temp_sort(left, left_pos-1)
            temp_sort(right_pos+1, right)  # 注意refer在的位置是已经确定的
        temp_sort(0, len(nums)-1)
        return(nums)
        '''
        # 快排实现二：关键是将数组分成三段，其中[left+1,mid)是小于部分，[mid,k)是大于部分，而[k,right)是未遍历部分
        def temp_sort(left, right):
            if right-left == 0:
                return
            refer = nums[left]
            midindex = left+1
            for k in range(left+1, right):
                if nums[k] < refer:  # 滚动大于的部分
                    nums[k], nums[midindex] = nums[midindex], nums[k]
                    midindex += 1
            nums[left], nums[midindex-1] = nums[midindex-1], nums[left]  # 最后需要将参考结点放置到该有的位置
            temp_sort(left, midindex-1)  # 注意参考点一定是放好位置的
            temp_sort(midindex, right)
        temp_sort(0, len(nums))
        return(nums)
        '''
        '''
        # 堆排序，注意从小到大排序需要建立大顶堆
        def down_filter(temp_index, heap_lenght):  # 建堆时需要确定堆的大小
            left_c = temp_index*2+1
            right_c = temp_index*2+2
            if left_c >= heap_lenght and right_c >= heap_lenght:
                return
            else:
                if right_c >= heap_lenght:
                    choose = left_c
                else:
                    choose = left_c if nums[left_c] > nums[right_c] else right_c
                if nums[choose] > nums[temp_index]:
                    nums[choose], nums[temp_index] = nums[temp_index], nums[choose]
                    down_filter(choose, heap_lenght)  # 递归
        for k in range(len(nums)-1, -1, -1):  # 从下往上建堆更快
            down_filter(k, len(nums))
        for k in range(len(nums)-1, -1, -1):
            nums[0], nums[k] = nums[k], nums[0]
            down_filter(0, k)
        return(nums)
        '''

    def trap(self, height):
        # 接雨水，关键是将雨水分成两种考虑，一种是靠着右边墙面的，一种是靠着左边墙面的
        if len(height) == 0:
            return 0
        result = 0
        refer_left = height[0]
        temp_result = 0
        for index in range(1, len(height)):
            if height[index] <= refer_left:
                temp_result += (refer_left-height[index])
            else:
                refer_left = height[index]
                result += temp_result
                temp_result = 0
        temp_result = 0
        refer_right = height[len(height)-1]
        for index in range(len(height)-2, -1, -1):
            if height[index] < refer_right:
                temp_result += (refer_right-height[index])
            else:
                refer_right = height[index]
                result += temp_result
                temp_result = 0
        return result

    def maximumGap(self, nums):
        # 数组中的最大间隔，使用桶分割元素快速计算
        if len(nums) < 2:
            return 0
        max_num, min_num = max(nums), min(nums)
        box_size = max((max_num-min_num)//(len(nums)-1), 1)  # 选取box大小，确保n个box涵盖数值区域，且最后一个桶一定涵盖最后一个元素，防止取0
        box_num = (max_num-min_num)//box_size+1  # 这种情况下桶的个数会超过n
        box_list = [[] for _ in range(box_num)]
        for num in nums:
            box_index = (num-min_num)//box_size
            box_list[box_index].append(num)
        new_list = [box for box in box_list if box]  # 剔除空元素
        result = 0
        for box_index in range(1, len(new_list)):
            result = max(result, min(new_list[box_index])-max(new_list[box_index-1]))  # 鸽巢原理可以保证两个box之间的差异大于box内部的差异
        return result

    def searchRange(self, nums, target):
        # 排序数组定位某一个元素的边界，采用二分查找
        # 首先明确问题的定义，找到大于目标结点的最小index，找到小于目标结点得最大index，注意如果直接找相等的边界不好定义
        # 明确定义后需要处理边界条件
        nums = [-sys.maxsize-1]+nums+[sys.maxsize]

        def find_right(left, right):  # 在范围内返回大于target的最小idnex
            if left == right:
                return left
            mid = (left+right)//2
            if nums[mid] <= target:
                return find_right(mid+1, right)
            else:
                return find_right(left, mid)

        def find_left(left, right):
            if left == right:
                return left
            mid = (left+right+1)//2  # 注意此时取上整
            if nums[mid] >= target:
                return find_left(left, mid-1)
            else:
                return find_left(mid, right)

        right_index = find_right(0, len(nums)-1)
        left_index = find_left(0, len(nums)-1)
        return [left_index, right_index-2] if right_index-left_index > 1 else [-1, -1]


if __name__ == '__main__':
    solu = Solution()
    # result = solu.sortArray([5, 2, 3, 1])
    # result = solu.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    # result = solu.maximumGap([1, 1, 1, 1])
    result = solu.searchRange([5, 7, 7, 8, 8, 10], 8)
    pass
