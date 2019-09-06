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
        result = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                if i != j:
                    temp_num = nums[i] ^ nums[j]
                    temp_str = bin(temp_num).replace('0b', '')
                    result += temp_str.count('1')
        return result

    def isInterleave(self, s1, s2, s3):
        '''        
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
        def check(s1_temp, s2_temp, s3_temp):
            cut_3=len(s3_temp)//2
            for 
        dp_s1 = 0
        pass


'''
def fun():
    fancy_loading = deque('>--------------------')
    while True:
        print(str().join(fancy_loading), end='\r')  # 注意\r代表在这一行重复输出
        fancy_loading.rotate(1)
        time.sleep(0.1)
'''

if __name__ == '__main__':
    solu = Solution()
    # solu.minDistance('abcde', 'bcda')
    # result = solu.findMedianSortedArrays([1, 2, 3, 4, 5, 6], [2, 8, 9, 10, 11, 12])
    # result = solu.maxProduct([2, 0, 1, 3, -2])
    # result = solu.totalHammingDistance([4, 14, 2])
    result = solu.isInterleave('a', 'b', 'a')
