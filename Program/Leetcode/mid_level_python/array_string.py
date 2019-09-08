import sys


class Solution:
    def threeSum(self, nums):
        unique = list(set(nums))
        unique.sort()
        count_list = list(map(nums.count, unique))
        result = list()
        for begin in range(len(unique)):
            second = begin
            end = len(unique)-1
            while(second <= end):
                temp_sum = unique[begin]+unique[second]+unique[end]
                if temp_sum == 0:
                    pos_list = [begin, second, end]
                    if pos_list.count(begin) <= count_list[begin] and pos_list.count(second) <= count_list[second] and pos_list.count(end) <= count_list[end]:
                        result.append([unique[begin], unique[second], unique[end]])
                    second += 1
                    end -= 1
                elif temp_sum > 0:
                    end -= 1
                else:
                    second += 1
        return result

    def setZeroes(self, matrix):
        shape = [len(matrix), len(matrix[0])]
        row_have, col_have = False, False
        for row_index in range(shape[0]):
            if matrix[row_index][0] == 0:
                col_have = True
        for col_index in range(shape[1]):
            if matrix[0][col_index] == 0:
                row_have = True
        for row_index in range(shape[0]):
            for col_index in range(shape[1]):
                if matrix[row_index][col_index] == 0:
                    matrix[row_index][0] = 0
                    matrix[0][col_index] = 0
        for row_index in range(1, shape[0]):
            for col_index in range(1, shape[1]):
                if matrix[row_index][0] == 0 or matrix[0][col_index] == 0:
                    matrix[row_index][col_index] = 0
        if row_have:
            for col_index in range(shape[1]):
                matrix[0][col_index] = 0
        if col_have:
            for raw_index in range(shape[0]):
                matrix[raw_index][0] = 0

    def groupAnagrams(self, strs):
        the_dict = dict()
        for elem in strs:
            sorted_elem = ''.join(sorted(elem))  # python中队str进行排序
            if the_dict.get(sorted_elem):
                the_dict[sorted_elem].append(elem)
            else:
                the_dict[sorted_elem] = list([elem])
        result = list()
        for dic in the_dict:
            result.append(the_dict[dic])
        return result

    def lengthOfLongestSubstring(self, s):  # 关键思想是滑动窗口以及每次的重定位
        the_dict = dict()
        res, left = 0, 0
        for right in range(len(s)):
            if the_dict.get(s[right]) is not None:
                left = max(the_dict[s[right]]+1, left)
            the_dict[s[right]] = right
            res = max(res, right-left+1)
        return res

    def longestPalindrome(self, s):
        # 马拉车算法
        '''
        expand_s = '$#' + '#'.join(list(s))+'#%'
        len_s = [0]*len(expand_s)
        max_right, max_middle = 0, 0
        result_len, result_mid = 0, 0
        for index_s in range(1, len(expand_s)-1):
            if index_s > max_right:
                len_s[index_s] = 1
            else:
                len_s[index_s] = min(len_s[2*max_middle-index_s], max_right-index_s+1)
            while(expand_s[index_s+len_s[index_s]] == expand_s[index_s-len_s[index_s]]):
                len_s[index_s] += 1
            if len_s[index_s]+index_s-1 > max_right:
                max_right = len_s[index_s]+index_s-1
                max_middle = index_s
            if result_len < len_s[index_s]:
                result_len = len_s[index_s]
                result_mid = index_s
        temp_res = expand_s[result_mid-result_len+1: result_mid+result_len]
        final_res = temp_res.replace('#', '')
        return final_res
        '''

    def increasingTriplet(self, nums):
        small_1, small_2 = sys.maxsize, sys.maxsize
        for num in nums:
            if num > small_2:
                return True
            elif num > small_1:
                small_2 = num
            else:
                small_1 = num
        return False


if __name__ == '__main__':
    solu = Solution()
    int_list = [-2, 0, 0, 2, 2]
    int_matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    str_list = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
    string = "a"
    # result = solu.threeSum(int_list)
    # solu.setZeroes(int_matrix)
    # result = solu.groupAnagrams(str_list)
    # result = solu.lengthOfLongestSubstring(string)
    # result = solu.longestPalindrome(string)
    result = solu.increasingTriplet(int_list)
    print('end')
