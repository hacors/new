#include <vector>
#include <map>
#include <set>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "__support.h"
using namespace std;
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution
{
  public:
    void deleteNode(ListNode *node)
    {
        node->val = node->next->val;
        ListNode *temp = node->next;
        node->next = temp->next;
        delete temp;
    }
    ListNode *removeNthFromEnd(ListNode *head, int n)
    {
        ListNode *blankhead = new ListNode(0);
        blankhead->next = head;
        ListNode *pre = blankhead;
        ListNode *totail = blankhead;
        while (n--)
            totail = totail->next;
        while (totail->next != NULL)
        {
            totail = totail->next;
            pre = pre->next;
        }
        ListNode *deleted = pre->next;
        pre->next = deleted->next;
        delete deleted;
        return blankhead->next;
    }
    ListNode *reverseList(ListNode *head)
    {
        //迭代思路
        /*
        ListNode *temp;  
        ListNode *pos = head;
        ListNode *res = new ListNode(0);
        while (pos != NULL)
        {
            temp = pos;
            pos = pos->next;
            temp->next = res->next;
            res->next = temp;
        }
        return res->next;
        */
        if (head == NULL || head->next == NULL)
            return head;
        else
        {
            ListNode *newhead = reverseList(head->next);
            head->next->next = head;
            head->next = NULL;
            return newhead;
        }
    }
    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
    {
        ListNode *pos1 = l1;
        ListNode *pos2 = l2;
        ListNode *choice, *end;
        ListNode *result = new ListNode(0);
        end = result;
        while (pos1 != NULL && pos2 != NULL)
        {
            if (pos1->val >= pos2->val)
            {
                choice = pos2;
                pos2 = pos2->next;
            }
            else
            {
                choice = pos1;
                pos1 = pos1->next;
            }
            choice->next = NULL;
            end->next = choice;
            end = choice;
        }
        if (pos1 != NULL)
            end->next = pos1;
        else
            end->next = pos2;
        return result->next;
    }
    bool isPalindrome(ListNode *head)
    {
        if (head == NULL || head->next == NULL)
            return true;
        ListNode *cut = head, *theend = head;
        while (theend)
        {
            cut = cut->next;
            theend = theend->next ? theend->next->next : nullptr;
        }
        cut = reverseList(cut);
        while (cut != NULL)
        {
            if (cut->val != head->val)
                return false;
            cut = cut->next;
            head = head->next;
        }
        return true;
    }
    bool hasCycle(ListNode *head)
    {
        if (head == NULL || head->next == NULL)
            return false;
        ListNode *fast = head, *slow = head;
        while (fast)
        {
            slow = slow->next;
            fast = fast->next ? fast->next->next : NULL;
            if (fast == slow)
                return true;
        }
        return false;
    }
    ListNode *_startlist(vector<int> &vec)
    {
        ListNode *head = new ListNode(0);
        ListNode *pos = head;
        int len = vec.size();
        for (int i = 0; i < len; i++)
        {
            pos->next = new ListNode(vec[i]);
            pos = pos->next;
        }
        return head->next;
    }
    void _printlist(ListNode *list)
    {
        while (list != NULL)
        {
            cout << list->val;
            list = list->next;
        }
        cout << endl;
    }
};
int main()
{
    Solution mysolution;
    vector<int> temp1 = {1, 2, 2, 1};
    vector<int> temp2 = {2};
    ListNode *list1 = mysolution._startlist(temp1);
    ListNode *list2 = mysolution._startlist(temp2);
    cout << mysolution.isPalindrome(list1);
    return 0;
}