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


if __name__ == '__main__':
    solu = Solution()
    result = solu.sortArray([5, 2, 3, 1])
    pass
