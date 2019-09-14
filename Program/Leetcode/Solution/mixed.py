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

    def numPermsDISequence(self, S):
        # 给出整数符合一定大小顺序要求的所有排列方式

        pass


if __name__ == '__main__':
    solu = Solution()
    result = solu.findMedianSortedArrays([10000], [10001])
    pass
