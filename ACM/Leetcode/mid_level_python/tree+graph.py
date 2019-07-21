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
    def midorderTraversal(self, root):
        ''''
        左孩子入栈，出栈转右孩子
        '''
        result = []
        my_stack = list()
        temp_pos = root
        while True:
            while temp_pos:
                my_stack.append(temp_pos)
                temp_pos = temp_pos.left
            if len(my_stack) == 0:
                break
            temp_pos = my_stack.pop()
            result.append(temp_pos.val)
            temp_pos = temp_pos.right
        return result

    def preorderTraversal(self, root):
        '''
        左下访问，右孩子入栈
        '''
        '''
        my_stack = list()
        temp_pos = root
        while True:
            while temp_pos:
                print(temp_pos.val)
                my_stack.append(temp_pos.right)
                temp_pos = temp_pos.left
            if len(my_stack) == 0:
                break
            temp_pos = my_stack.pop()
        '''
        result = []
        my_stack = [root]
        while len(my_stack):
            temp_pos = my_stack.pop()
            result.append(temp_pos.val)
            if temp_pos.right:
                my_stack.append(temp_pos.right)
            if temp_pos.left:
                my_stack.append(temp_pos.left)
        return result

    def postorderTraversal(self, root):
        '''
        树的左边的数据链，并且这个左边链的每个结点的右兄弟都在其栈前
        每次出栈都应该判断是不是兄弟，如果是兄弟就需要继续找左边链
        '''
        result = []
        my_stack = []
        my_stack.append(root)
        temp_pos = root
        while len(my_stack):
            if temp_pos != my_stack[-1].right and temp_pos != my_stack[-1].left:
                temp_pos = my_stack[-1]  # 非直系就给兄弟机会
                while temp_pos:
                    if temp_pos.left:
                        if temp_pos.right:
                            my_stack.append(temp_pos.right)
                        my_stack.append(temp_pos.left)
                        temp_pos = temp_pos.left
                    else:
                        my_stack.append(temp_pos.right)
                        temp_pos = temp_pos.right
                my_stack.pop()  # 弹出最后无效的右孩子
            temp_pos = my_stack.pop()
            result.append(temp_pos.val)
        return result

    def levelorderTraversal(self, root):
        result = []
        my_queue = [root]
        while len(my_queue):
            temp_pos = my_queue.pop(0)
            result.append(temp_pos.val)
            if temp_pos.left:
                my_queue.append(temp_pos.left)
            if temp_pos.right:
                my_queue.append(temp_pos.right)
        return result

    def zigzagLevelOrder(self, root):
        result = []
        if not root:
            return result
        my_queue_left = [root]
        my_queue_right = []
        while len(my_queue_left) or len(my_queue_right):
            if len(my_queue_left):
                result.append([])
                while len(my_queue_left):
                    temp_pos = my_queue_left.pop(0)
                    result[-1].append(temp_pos.val)
                    if temp_pos.left:
                        my_queue_right.append(temp_pos.left)
                    if temp_pos.right:
                        my_queue_right.append(temp_pos.right)
            if len(my_queue_right):
                result.append([])
                while len(my_queue_right):
                    temp_pos = my_queue_right.pop()
                    result[-1].append(temp_pos.val)
                    if temp_pos.right:
                        my_queue_left.insert(0, temp_pos.right)
                    if temp_pos.left:
                        my_queue_left.insert(0, temp_pos.left)
        return result

    def buildTree(self, preorder, inorder):
        if len(preorder):
            temp_root = TreeNode(preorder[0])
            index_root = inorder.index(preorder[0])
            temp_root.left = Solution.buildTree(self, preorder[1:index_root+1], inorder[0:index_root])
            temp_root.right = Solution.buildTree(self, preorder[1+index_root:], inorder[index_root+1:])
            return temp_root

    def connect(self, root):
        pass


if __name__ == '__main__':
    solu = Solution()
    temp_tree = get_tree([1, 2, 3, 4, None, None, 5, None, 6, None, None, 7, 8])
    pre_list = solu.preorderTraversal(temp_tree)
    mid_list = solu.midorderTraversal(temp_tree)
    # solu.postorderTraversal(temp_tree)
    # solu.levelorderTraversal(temp_tree)
    # solu.zigzagLevelOrder(temp_tree)
    # build_tree = solu.buildTree(pre_list, mid_list)
    solu.connect(temp_tree)
    pass
