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

    def solveSudoku(self, board):
        # 数独游戏
        '''
        pos_list = []
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    pos_list.append([i, j])

        def check(temp_board, pos):
            num = temp_board[pos[0]][pos[1]]
            for index in range(9):
                if (index != pos[1] and temp_board[pos[0]][index] == num) or\
                        (index != pos[0] and temp_board[index][pos[1]] == num):
                    return False
            big_i, big_j = pos[0]//3, pos[1]//3
            for index_i in range(3):
                for index_j in range(3):
                    if (index_i != pos[0] % 3 and index_j != pos[1] % 3) and temp_board[big_i*3 + index_i][big_j*3+index_j] == num:
                        return False
            return True

        def fill(temp_board, pos_list):
            if len(pos_list) == 0:
                return True
            for candidate in range(1, 10):
                temp_pos = pos_list.pop()
                temp_board[temp_pos[0]][temp_pos[1]] = str(candidate)
                if check(temp_board, temp_pos) and fill(temp_board, pos_list):
                    return True
                pos_list.append(temp_pos)
                temp_board[temp_pos[0]][temp_pos[1]] = '.'  # 这是一个关键点，回溯法需要保持状态的恢复，这个恢复需要涉及到所有的参数的恢复
            return False
        fill(board, pos_list)
        '''
        row, col, box = [], [], []
        need_fill = []
        for temp in range(9):
            row.append(set(range(1, 10)))
            col.append(set(range(1, 10)))
            box.append(set(range(1, 10)))
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    row[i].remove(num)
                    col[j].remove(num)
                    box[i//3*3+j//3].remove(num)
                else:
                    need_fill.append([i, j])

        def fill(temp_need_fill, temp_row, temp_col, temp_box):
            if len(temp_need_fill) == 0:
                return True
            else:
                temp_i, temp_j = temp_need_fill[-1]
                for temp_choose in temp_row[temp_i] & temp_col[temp_j] & temp_box[temp_i//3*3+temp_j//3]:
                    temp_need_fill.pop()  # 关键要在循环里面实现状态的更新和恢复
                    board[temp_i][temp_j] = str(temp_choose)
                    temp_row[temp_i].remove(temp_choose)
                    temp_col[temp_j].remove(temp_choose)
                    temp_box[temp_i//3*3+temp_j//3].remove(temp_choose)
                    if fill(temp_need_fill, temp_row, temp_col, temp_box):
                        return True
                    # temp_board[temp_i][temp_j] = '.'
                    temp_row[temp_i].add(temp_choose)
                    temp_col[temp_j].add(temp_choose)
                    temp_box[temp_i//3*3+temp_j//3].add(temp_choose)
                    temp_need_fill.append([temp_i, temp_j])  # 关键
                return False
        fill(need_fill, row, col, box)


if __name__ == '__main__':
    solu = Solution()
    arr = list([
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ])
    result = solu.solveSudoku(arr)
    pass
