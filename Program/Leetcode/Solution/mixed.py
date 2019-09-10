class Solution():
    def findKthNumber(self, m, n, k):
        # 寻找乘法表中第k小的数
        def enough(x):  # 统计小于等于这个数的所有数目（如果这个数目满足要求，那么返回TRUE）
            count = sum(min(x//i, n) for i in range(1, m+1))
            return count >= k
        left = 1
        right = m*n
        while left < right:
            mid = (left+right)//2
            if enough(mid):
                right = mid
            else:
                left = mid+1
        return left

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

    def totalHammingDistance(self, nums):
        '''
        result = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                if i != j:
                    temp_num = nums[i] ^ nums[j]
                    temp_str = bin(temp_num).replace('0b', '')
                    result += temp_str.count('1')
        return result
        '''
        # 思考：既然都是逐位运算，那么可以考虑将所有的位分开看待，这样可能存在重复的计算模式
        if not nums:
            return 0
        size = len(bin(max(nums))[2:])
        result = 0
        while size:
            temp_count = 0
            for i in range(len(nums)):
                if nums[i] % 2:
                    temp_count += 1
                nums[i] = nums[i]//2
            result += temp_count*(len(nums)-temp_count)
            size -= 1
        return result

    def diffWaysToCompute(self, input):
        # 给一个计算式添加括号优先级，得出所有的计算结果
        # 拆分
        num_list = []
        symbol_list = []
        temp = ''
        for char in input:
            if char in ['+', '*', '-']:
                num_list.append(int(temp))
                temp = ''
                symbol_list.append(char)
            else:
                temp += char
        num_list.append(int(temp))

        def posible_list(num_list, symbol_list):
            posible_result = []
            if not symbol_list:  # 如果只剩下一个数，那么只有一种情况
                posible_result.append(num_list[0])
            else:
                for symbol_index in range(len(symbol_list)):  # 否则遍历所有的符号，找出所有的可能
                    left_posible = posible_list(num_list[:symbol_index+1], symbol_list[:symbol_index])  # 这部分是符号左侧所有的可能
                    right_posible = posible_list(num_list[symbol_index+1:], symbol_list[symbol_index+1:])  # 这部分是符号右侧所有的可能
                    the_symbol = symbol_list[symbol_index]
                    if the_symbol == '+':
                        posible_result.extend(left+right for left in left_posible for right in right_posible)
                    elif the_symbol == '*':
                        posible_result.extend(left*right for left in left_posible for right in right_posible)
                    else:
                        posible_result.extend(left-right for left in left_posible for right in right_posible)
            return posible_result
        return posible_list(num_list, symbol_list)

    def numPermsDISequence(self, S):
        # 给出整数符合一定大小顺序要求的所有排列方式

        pass


if __name__ == '__main__':
    solu = Solution()
    result = solu.findMedianSortedArrays([10000], [10001])
    pass
