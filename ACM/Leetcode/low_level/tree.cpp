#include <vector>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "__support.h"
using namespace std;
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution
{
  public:
    int maxDepth(TreeNode *root)
    {
        if (root == NULL)
            return 0;
        else if (root->left == NULL && root->right == NULL)
            return 1;
        else
            return (max(maxDepth(root->left), maxDepth(root->right)) + 1);
    }
    long long _max = -__LONG_LONG_MAX__ - 1;
    bool isValidBST(TreeNode *root)
    {
        /*
        if (root == NULL)
            return true;
        if (root->left == NULL && root->right == NULL)
            return true;
        long long now, pre = -__LONG_LONG_MAX__ - 1;
        stack<TreeNode *> tempstack;
        TreeNode *temproot = root;
        while (true)
        {
            while (temproot)
            {
                tempstack.push(temproot);
                temproot = temproot->left;
            }
            if (tempstack.empty())
                break;
            temproot = tempstack.top();
            tempstack.pop();
            now = temproot->val;
            if (now <= pre)
                return false;
            pre = now;
            temproot = temproot->right;
        }
        return true;
        */
        if (root == NULL)
        {
            return true;
        }
        if (isValidBST(root->left))
        {
            if (_max < root->val)
            {
                _max = root->val;
                return isValidBST(root->right);
            }
        }
        return false;
    }
    bool flag = true;
    bool isSymmetric(TreeNode *root)
    {
        /*
        if (root == NULL)
            return true;
        if (root->left == NULL && root->right == NULL)
            return true;
        root->right = _reversetree(root->right);
        _issametree(root->left, root->right);
        return flag;
        */
        if (root == NULL)
            return true;
        return isSymmetric(root->left, root->right);
    }
    bool isSymmetric(TreeNode *left, TreeNode *right)
    {
        if (left == NULL && right == NULL)
            return true;
        else if ((left != NULL && right == NULL) || (left == NULL && right != NULL))
            return false;
        else if (left->val != right->val)
            return false;
        else
            return isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
    }
    TreeNode *_reversetree(TreeNode *root)
    {
        if (root == NULL)
            return root;
        root->left = _reversetree(root->left);
        root->right = _reversetree(root->right);
        TreeNode *temp = root->left;
        root->left = root->right;
        root->right = temp;
        return root;
    }
    void _issametree(TreeNode *tree1, TreeNode *tree2)
    {
        if (tree1 == tree2)
            return;
        if ((tree1 == NULL && tree2 != NULL) || (tree2 == NULL && tree1 != NULL))
        {
            flag = false;
            return;
        }
        if (tree1->val != tree2->val)
        {
            flag = false;
            return;
        }
        _issametree(tree1->left, tree2->left);
        _issametree(tree1->right, tree2->right);
    }
    vector<vector<int>> levelOrder(TreeNode *root)
    {
        vector<vector<int>> result;
        if (root == NULL)
            return result;
        //设定变量
        vector<TreeNode *> temp;
        temp.push_back(root);
        int cur = 0, last = 1, level = 0;
        while (cur < temp.size())
        {
            last = temp.size();
            vector<int> newd;
            result.push_back(newd);
            while (cur < last)
            {
                result[level].push_back(temp[cur]->val);
                if (temp[cur]->left)
                    temp.push_back(temp[cur]->left);
                if (temp[cur]->right)
                    temp.push_back(temp[cur]->right);
                cur++;
            }
            level++;
        }
        return result;
    }
    TreeNode *sortedArrayToBST(vector<int> &nums)
    {
        if (nums.empty())
            return NULL;
        return sortArrayToBSTloop(nums, 0, nums.size() - 1);
    }
    TreeNode *sortArrayToBSTloop(vector<int> &nums, int start, int end)
    {
        if (start > end)
            return NULL;
        int mid = start + (end - start) / 2;
        TreeNode *root = new TreeNode(nums[mid]);
        root->left = sortArrayToBSTloop(nums, start, mid - 1);
        root->right = sortArrayToBSTloop(nums, mid + 1, end);
        return root;
    }
    void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
    {
        int mp = m - 1, np = n - 1, sp = m + n - 1;
        while (mp >= 0 && np >= 0)
        {
            if (nums1[mp] >= nums2[np])
            {
                nums1[sp] = nums1[mp];
                mp--;
                sp--;
            }
            else
            {
                nums1[sp] = nums2[np];
                np--;
                sp--;
            }
        }
        if (mp < 0)
        {
            while (np >= 0)
            {
                nums1[sp] = nums2[np];
                np--;
                sp--;
            }
        }
    }
    int firstBadVersion(int n){return firstBadVersionloop(1, n)};
    int firstBadVersionloop(int begin, int end)
    {
        if (begin == end)
            return begin;
        int cut = begin + (end - begin) / 2;
        if (isBadVersion(cut))
            return firstBadVersionloop(begin, cut);
        else
            return firstBadVersionloop(cut + 1, end);
    }
    bool isBadVersion(int version){};
    TreeNode *_starttree(vector<int> &vec)
    {
        vec.insert(vec.begin(), {0, -1});
        TreeNode *fakeroot = new TreeNode(0), *tempnode;
        queue<TreeNode *> tempqueue;
        int tempint;
        tempqueue.push(fakeroot);
        int pos = 1, templeft, tempright;
        while (!tempqueue.empty())
        {
            tempnode = tempqueue.front();
            tempqueue.pop();
            templeft = pos < vec.size() ? vec[pos] : -1;
            tempright = pos + 1 < vec.size() ? vec[pos + 1] : -1;
            pos = pos + 2;
            if (templeft != -1)
            {
                tempnode->left = new TreeNode(templeft);
                tempqueue.push(tempnode->left);
            }
            if (tempright != -1)
            {
                tempnode->right = new TreeNode(tempright);
                tempqueue.push(tempnode->right);
            }
        }
        return fakeroot->right;
    }
};
int main()
{
    Solution mysolution;
    vector<int> nums1 = {};
    vector<int> nums2 = {1};
    int m = 0, n = 1;
    mysolution.merge(nums1, 0, nums2, 1);
    return 0;
}