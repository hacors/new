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
        pass


if __name__ == '__main__':
    solu = Solution()
    int_list = [-2, 0, 0, 2, 2]
    int_matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    # result = solu.threeSum(int_list)
    solu.setZeroes(int_matrix)
    print('end')
