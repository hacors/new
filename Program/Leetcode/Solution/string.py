class Solution():
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


if __name__ == '__main__':
    solu = Solution()
    pass
