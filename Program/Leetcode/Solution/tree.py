import queue


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None


def get_tree(tree_num):
    new_tree_num = [0, None]+tree_num+[None]*len(tree_num)
    fake_root = TreeNode(0)
    pos_queue = queue.Queue()
    pos_queue.put(fake_root)
    num_index = 1
    while not pos_queue.empty():
        temp_node = pos_queue.get()
        if new_tree_num[num_index]:
            temp_node.left = TreeNode(new_tree_num[num_index])
            pos_queue.put(temp_node.left)
        if new_tree_num[num_index+1]:
            temp_node.right = TreeNode(new_tree_num[num_index+1])
            pos_queue.put(temp_node.right)
        num_index += 2
    return fake_root.right


class Solution:
    def lowestCommonAncestor(self, root, p, q):
        pass


if __name__ == '__main__':
    solu = Solution()
    temp_tree = get_tree([5, 3, 7, 2, 4, 6, 8, 1, None])
    
    result = solu.kthSmallest(temp_tree, 6)
    pass
