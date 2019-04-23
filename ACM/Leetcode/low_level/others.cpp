#include <stdlib.h>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "__support.h"
using namespace std;
class Solution
{
  public:
    int hammingWeight(uint32_t n)
    {
        /*
        if (n == 0)
            return 0;
        else if (n % 2 == 0)
            return hammingWeight(n / 2);
        else
            return hammingWeight(n / 2) + 1;
        */
        int ans = 0;
        while (n)
        {
            ans += n % 2;
            n >>= 1;
        }
        return ans;
    }
    int hammingDistance(int x, int y)
    {
        /*
        int res = 0;
        while (x != 0 || y != 0)
        {
            if (x % 2 != y % 2)
                res++;
            x = x / 2;
            y = y / 2;
        }
        return res;
        */
        int k = x ^ y;
        int res = 0;
        while (k)
        {
            if (k & 1)
                res++;
            k >>= 1;
        }
        return res;
    }
    uint32_t reverseBits(uint32_t n)
    {
        uint32_t res = 0;
        int pos = 0;
        while (n)
        {
            if (n & 1)
                res += pow(2, 31 - pos);
            pos++;
            n >>= 1;
        }
        return res;
    }
    vector<vector<int>> generate(int numRows)
    {
        vector<vector<int>> res;
        if (numRows == 0)
            return res;
        res = {{1}};
        for (int i = 1; i < numRows; i++)
        {
            vector<int> temp = {1};
            for (int j = 0; j < res[i - 1].size() - 1; j++)
                temp.push_back(res[i - 1][j] + res[i - 1][j + 1]);
            temp.push_back(1);
            res.push_back(temp);
        }
        return res;
    }
    bool isValid(string s)
    {
        stack<char> store;
        map<char, char> match = {{'(', ')'}, {'[', ']'}, {'{', '}'}};
        for (int i = 0; i < s.length(); i++)
        {
            if (!store.empty() && s[i] == match[store.top()])
                store.pop();
            else
                store.push(s[i]);
        }
        if (store.empty())
            return true;
        else
            return false;
    }
    int missingNumber(vector<int> &nums)
    {
        /*
        int max = 0;
        int result = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            if (nums[i] > max)
                max = nums[i];
            result = result ^ nums[i];
        }
        for (int i = 0; i <= max; i++)
            result = result ^ i;
        if (max == nums.size()-1)
            return (nums.size());
        else
            return result;
        */
        int sums = 0, checksum = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            sums += nums[i];
            checksum += i + 1;
        }
        return checksum - sums;
    }
};
int main()
{
    Solution mysolu;
    vector<int> test{1, 0, 3};
    mysolu.missingNumber(test);
    return 0;
}