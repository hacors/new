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
    def addTwoNumbers(self, l1, l2):
        result = ListNode(None)
        cur = result
        calculate = 0
        while l1 or l2 or calculate:
            if l1:
                calculate = l1.val+calculate
                l1 = l1.next
            if l2:
                calculate = l2.val+calculate
                l2 = l2.next
            cur.next = ListNode(calculate % 10)
            cur = cur.next
            calculate = int(calculate/10)
        return result.next

    def oddEvenList(self, head):
        result_odd = ListNode(None)
        cur_odd = result_odd
        result_even = ListNode(None)
        cur_even = result_even
        index = 1
        cur_origin = head
        while cur_origin:
            if index % 2:
                cur_odd.next = ListNode(cur_origin.val)
                cur_odd = cur_odd.next
            else:
                cur_even.next = ListNode(cur_origin.val)
                cur_even = cur_even.next
            cur_origin = cur_origin.next
            index += 1
        cur_odd.next = result_even.next
        return result_odd.next

    def getIntersectionNode(self, headA, headB):
        if not (headA and headB):
            return None
        pA = headA
        pB = headB
        while pA != pB:  # 跳两次
            if pA is None:
                pA = headB
            else:
                pA = pA.next
            if pB is None:
                pB = headA
            else:
                pB = pB.next
        return pA
        '''
        cur_a = headA
        cur_b = headB
        while cur_a and cur_b:
            cur_a = cur_a.next
            cur_b = cur_b.next
        if cur_a:
            temp_a = headA
            while cur_a:
                temp_a = temp_a.next
                cur_a = cur_a.next
            cur_a = temp_a
            cur_b = headB
        else:
            temp_b = headB
            while cur_b:
                temp_b = temp_b.next
                cur_b = cur_b.next
            cur_b = temp_b
            cur_a = headA
        while cur_a and cur_b:
            if cur_a == cur_b:
                return cur_a
            else:
                cur_a = cur_a.next
                cur_b = cur_b.next
        return None
        '''


if __name__ == '__main__':
    solu = Solution()
    list_a = get_list([8, 2, 3])
    list_b = get_list([8, 0, 4, 3])
    # solu.addTwoNumbers(list_a, list_b)
    result = solu.oddEvenList(list_a)
    result = solu.getIntersectionNode(list_a, list_b)
    pass
