from collections import deque
import time


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
    solu.minDistance('abcde', 'bcda')
