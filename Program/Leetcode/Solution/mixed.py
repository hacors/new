class Solution():
    def minDistance(self, word1, word2):
        '''
        本质，对于两个末尾字符相同的子序列，那么这两个序列的距离就是分别忽略末尾字符后的距离
        对于两个末尾字符不同的子序列，那么要充分考虑分别的倒数第二位字符的匹配目标
        '''
        first_d = len(word1)+1
        second_d = len(word2)+1
        dp = [[0 for j in range(second_d)] for i in range(first_d)]
        for i in range(first_d):
            dp[i][0] = i
        for j in range(second_d):
            dp[0][j] = j
        for i in range(1, first_d):
            for j in range(1, second_d):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])+1
        return dp[len(word1)][len(word2)]

    def findKthNumber(self, m, n, k):
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
        pass

    def maxProduct(self, nums):
        '''
        nums = [1]+nums
        dp_max = [1 for i in range(len(nums))]
        dp_min = [1 for i in range(len(nums))]
        for i in range(1, len(nums)):
            if nums[i] > 0:
                dp_max[i] = max(nums[i], dp_max[i-1]*nums[i])
                dp_min[i] = min(nums[i], dp_min[i-1]*nums[i])
            else:
                dp_max[i] = max(nums[i], dp_min[i-1]*nums[i])
                dp_min[i] = min(nums[i], dp_max[i-1]*nums[i])
        return max(dp_max[1:])
        '''
        left_temp = nums[0]
        right_temp = nums[-1]
        result = max(left_temp, right_temp)
        for i in range(1, len(nums)):
            left_temp = nums[i]*(left_temp or 1)
            right_temp = nums[len(nums)-1-i]*(right_temp or 1)
            result = max(result, left_temp, right_temp)
        return result

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

    def isInterleave(self, s1, s2, s3):
        '''
        # 深度优先搜索，朴素的思想
        if (len(s1)+len(s2)) != len(s3):
            return False

        def check(s1_temp, s2_temp, s3_temp):
            if len(s3_temp) == 0:
                return True
            else:
                temp_result = False
                if len(s1_temp) > 0 and s1_temp[0] == s3_temp[0]:
                    temp_result = check(s1_temp[1:], s2_temp, s3_temp[1:]) or temp_result
                if len(s2_temp) > 0 and s2_temp[0] == s3_temp[0]:
                    temp_result = check(s1_temp, s2_temp[1:], s3_temp[1:]) or temp_result
                return temp_result
        return check(s1, s2, s3)
        '''
        # 判断任何两段组合能不能形成形成新的组合，如果可以那么（不管他们怎么组合的，只需要判断有没有可能性）
        if len(s3) != len(s1)+len(s2):
            return False
        s1 = '**'+s1
        s2 = '**'+s2
        s3 = '****'+s3  # 去除边界判断
        dp_matrix = [[False for j in range(len(s2))]for i in range(len(s1))]
        for i in range(len(s1)):
            dp_matrix[i][0] = True
        for j in range(len(s2)):
            dp_matrix[0][j] = True
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                if (dp_matrix[i-1][j] and s1[i] == s3[i+j+1]) or (dp_matrix[i][j-1] and s2[j] == s3[i+j+1]):
                    dp_matrix[i][j] |= True
        return dp_matrix[len(s1)-1][len(s2)-1]

    def diffWaysToCompute(self, input):
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

    def superEggDrop(self, K, N):
        pass

    def findLength(self, A, B):  # 最长公共子数组
        # 使用二维dp算法，每个位置放置一个数，代表这个对应位置为结束两个数组的最长公共子数组
        A_bound = ['#']+A
        B_bound = ['#']+B
        dp_matrix = [[0 for j in range(len(B_bound))]for i in range(len(A_bound))]
        result = 0
        for i in range(1, len(A_bound)):
            for j in range(1, len(B_bound)):
                if A_bound[i] == B_bound[j]:
                    dp_matrix[i][j] = dp_matrix[i-1][j-1]+1
                    result = max(result, dp_matrix[i][j])
        return result

    def findTargetSumWays(self, nums, S):
        pass

    def red_bule(self, n, m):
        dp_matrix = [[0 for j in range(m+1)]for i in range(n+1)]


if __name__ == '__main__':
    solu = Solution()
    # solu.minDistance('abcde', 'bcda')
    # result = solu.findMedianSortedArrays([1, 2, 3, 4, 5, 6], [2, 8, 9, 10, 11, 12])
    # result = solu.maxProduct([2, 0, 1, 3, -2])
    # result = solu.totalHammingDistance([4, 14, 2, 3])
    # result = solu.isInterleave('a', 'b', 'a')
    # result = solu.diffWaysToCompute('3+44*5-6+1')
    # result = solu.superEggDrop(3, 100)
    # result = solu.findLength([1, 2, 3, 4], [3, 4, 1])
    pass
