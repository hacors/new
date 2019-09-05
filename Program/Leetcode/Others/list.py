class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def get_list(list_num):
    head = ListNode(None)
    cur = head
    for num in list_num:
        temp = ListNode(num)
        cur.next = temp
        cur = temp
    return head.next


class Solution:
    def mergeKLists(self, lists):
        
        pass


if __name__ == '__main__':
    solu = Solution()
    list_a = get_list([8, 2, 3])
    list_b = get_list([8, 0, 4, 3])
    list_c = get_list([0])
    list_d = get_list([2, 3, 5, 7, 8, -1])
    solu.mergeKLists([list_a, list_b, list_c, list_d])
    pass
