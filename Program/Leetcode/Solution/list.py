import heapq  # 本质是通过heapq帮助list操作元素，实现heap


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __lt__(self, other):
        return self.val < other.val


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
        # 多个有序链表的合并
        head = ListNode(-1)
        cursor = head
        the_heap = []
        for temp_list in lists:
            if temp_list:
                heapq.heappush(the_heap, (temp_list.val, temp_list))  # 关键在于每一个堆中的元素都可以记录下一次需要访问的位置
        while the_heap:
            temp_data, temp_node = heapq.heappop(the_heap)
            if temp_node.next:
                heapq.heappush(the_heap, (temp_node.next.val, temp_node.next))
            cursor.next = temp_node
            cursor = cursor.next
        return head.next

if __name__ == '__main__':
    solu = Solution()
    list_a = get_list([8, 2, 3])
    list_b = get_list([8, 0, 4, 3])
    list_c = get_list([0])
    list_d = get_list([2, 3, 5, 7, 8, -1])
    result = solu.mergeKLists([list_a, list_b, list_c, list_d])
    pass
