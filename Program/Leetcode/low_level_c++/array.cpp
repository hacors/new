#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include "__support.h"
using namespace std;
class Solution
{
public:
    int removeDuplicates(vector<int> &nums)
    {
        if (nums.size() == 0)
            return 0;
        int k = 1;
        for (int i = 1; i < nums.size(); i++)
        {
            if (nums[i - 1] != nums[i])
            {
                nums[k] = nums[i];
                k++;
            }
        }
        return k;
    }
    int maxProfit(vector<int> &prices)
    {
        if (prices.size() == 0)
            return 0;
        int sum = 0;
        for (int i = 1; i < prices.size(); i++)
        {
            if (prices[i] > prices[i - 1])
            {
                sum += (prices[i] - prices[i - 1]);
            }
        }
        return sum;
    }
    void rotate(vector<int> &nums, int k)
    {
        if (nums.empty() || (k %= nums.size()) == 0)
            return; //注意这是关键点
        int len = nums.size();
        _reverse(nums, 0, len - k - 1);
        _reverse(nums, len - k, len - 1);
        _reverse(nums, 0, len - 1);
    }
    void _reverse(vector<int> &nums, int l, int r)
    {
        int temp;
        int half = (r - l) / 2;
        for (int i = 0; i <= half; i++)
        {
            temp = nums[l + i];
            nums[l + i] = nums[r - i];
            nums[r - i] = temp;
        }
    }
    bool containsDuplicate(vector<int> &nums)
    {
        if (nums.size() <= 1)
            return false;
        sort(nums.begin(), nums.end());
        for (int i = 1; i < nums.size(); i++)
        {
            if (nums[i] == nums[i - 1])
                return true;
        }
        return false;
    }
    int singleNumber(vector<int> &nums)
    {
        int result = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            result = result ^ nums[i];
        }
        return result;
    }
    vector<int> intersect(vector<int> &nums1, vector<int> &nums2)
    {
        vector<int> temp;
        if (nums1.empty() || nums2.empty())
            return temp;
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        int p1 = 0;
        int p2 = 0;
        while (p1 < nums1.size() && p2 < nums2.size())
        {
            if (nums1[p1] == nums2[p2])
            {
                temp.push_back(nums1[p1]);
                p1++;
                p2++;
            }
            else
            {
                if (nums1[p1] > nums2[p2])
                    p2++;
                else
                    p1++;
            }
        }
        return temp;
    }
    vector<int> plusOne(vector<int> &digits)
    {
        int pos = digits.size() - 1;
        while (digits[pos] == 9)
        {
            digits[pos] = 0;
            pos--;
        }
        if (pos < 0)
            digits.insert(digits.begin(), 1);
        else
            digits[pos]++;
        return digits;
    }
    void moveZeroes(vector<int> &nums)
    {
        int len = nums.size();
        int viewed = 0;
        int numofnozero = 0;
        while (viewed < len)
        {
            if (nums[numofnozero] == 0)
            {
                viewed++;
                nums.erase(nums.begin() + numofnozero);
                nums.push_back(0);
            }
            else
            {
                viewed++;
                numofnozero++;
            }
        }
    }
    vector<int> twoSum(vector<int> &nums, int target)
    {
        vector<int> result;
        map<int, int> tempmap;
        for (int i = 0; i < nums.size(); i++)
        {
            int another = target - nums[i];
            if (tempmap.count(another))
            {
                result.push_back(tempmap[another]);
                result.push_back(i);
                return result;
            }
            else
                tempmap[nums[i]] = i;
        }
    }
    bool isValidSudoku(vector<vector<char>> &board)
    {
        /*
        {
        {'8', '3', 2, 2, '7', 2, 2, 2, 2}, 
        {'6', 2, 2, '1', '9', '5', 2, 2, 2}, 
        {2, '9', '8', 2, 2, 2, 2, '6', 2}, 
        {'8', 2, 2, 2, '6', 2, 2, 2, '3'}, 
        {'4', 2, 2, '8', 2, '3', 2, 2, '1'}, 
        {'7', 2, 2, 2, '2', 2, 2, 2, '6'}, 
        {2, '6', 2, 2, 2, 2, '2', '8', 2}, 
        {2, 2, 2, '4', '1', '9', 2, 2, '5'}, 
        {2, 2, 2, 2, '8', 2, 2, '7', '9'}
        };
        */
        set<char> row;
        set<char> col;
        set<char> block;
        for (int i = 0; i < 9; i++)
        {
            row.clear();
            col.clear();
            block.clear();
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] != 2)
                {
                    if (row.count(board[i][j]))
                        return false;
                    else
                        row.insert(board[i][j]);
                }
                if (board[j][i] != 2)
                {
                    if (col.count(board[j][i]))
                        return false;
                    else
                        col.insert(board[j][i]);
                }
                if (board[(i / 3) * 3 + j / 3][(i % 3) * 3 + j % 3] != 2)
                {
                    if (block.count(board[(i / 3) * 3 + j / 3][(i % 3) * 3 + j % 3]))
                        return false;
                    else
                        block.insert(board[(i / 3) * 3 + j / 3][(i % 3) * 3 + j % 3]);
                }
            }
        }
        return true;
    }
    void rotate(vector<vector<int>> &matrix)
    {
        int len = matrix.size();
        int rowlen = len / 2;
        int collen = (len + 1) / 2;
        int maxindex = len - 1;
        int temp = 0;
        for (int i = 0; i < rowlen; i++)
        {
            for (int j = 0; j < collen; j++)
            {
                temp = matrix[i][j];
                matrix[i][j] = matrix[maxindex - j][i];
                matrix[maxindex - j][i] = matrix[maxindex - i][maxindex - j];
                matrix[maxindex - i][maxindex - j] = matrix[j][maxindex - i];
                matrix[j][maxindex - i] = temp;
            }
        }
    }
    void _printvector(vector<int> &nums)
    {
        for (int i = 0; i < nums.size(); i++)
            cout << nums[i];
    }
};
int main()
{
    Solution mysolution;
    vector<int> test1 = {1, 2, 3, 4, 5, 6, 7};
    //vector<int> test2 = {2, 3, 4, 6, 2, 1};
    //vector<vector<char>> temp = {{5, 3, 8}, {6, 9, 7}, {2, 0, 1}};
    //bool result = mysolution.isValidSudoku(temp);
    vector<int> result = mysolution.twoSum(test1, 7);
    _printvector(result);
    return 0;
}