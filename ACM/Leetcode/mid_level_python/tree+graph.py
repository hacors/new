class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def get_tree(tree_num):
    new_tree_num = [0, None]+tree_num
    for num in tree_num:
        if num:
            temp_node=TreeNode(num)
            
    return


class Solution:
    def inorderTraversal(self, root):
        pass


if __name__ == '__main__':
    solu = Solution()
    tree_list = [1, None, 2, 3]
    get_tree(tree_list)
    pass
