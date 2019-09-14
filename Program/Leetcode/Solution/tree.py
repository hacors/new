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
        def find(temp_root, position_p, position_q):
            return True
        left_result = find(root.left, p, q)
        right_result = find(root.right, p, q)
        if root in [p, q] or (left_result and right_result):
            return root
        elif left_result:
            root.left
        pass


if __name__ == '__main__':
    solu = Solution()
    temp_tree = get_tree([5, 3, 7, 2, 4, 6, 8, 1, None, 10, 11])
    node_left = temp_tree.left.left.left
    node_right = temp_tree.left.right.left
    result = solu.lowestCommonAncestor(temp_tree, node_left, node_right)
    pass
