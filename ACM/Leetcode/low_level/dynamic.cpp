#include <vector>
#include <map>
#include <set>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "__support.h"
using namespace std;
class Solution
{
  public:
    int climbStairs(int n)
    {
        if (n == 0 || n == 1)
            return 1;
        vector<int> store = {1, 1};
        for (int i = 2; i < n; i++)
        {
            store.insert(store.end(), store[i - 1] + store[i - 2]);
            //_printvector(store);
        }
        return store[n - 1] + store[n - 2];
    }
    int maxProfit(vector<int> &prices)
    {
        int minprice = __INT_MAX__, maxpro = 0, size = prices.size();
        for (int i = 0; i < size; i++)
        {
            if (prices[i] < minprice)
                minprice = prices[i];
            if (prices[i] - minprice > maxpro)
                maxpro = prices[i] - minprice;
        }
        return maxpro;
    }
    int maxSubArray(vector<int> &nums)
    {
        vector<int> dp = {nums[0]};
        int result = nums[0];
        for (int i = 1; i < nums.size(); i++)
        {
            int temp = max(nums[i], dp[i - 1] + nums[i]);
            dp.push_back(temp);
            result = (temp > result) ? temp : result;
        }
        return result;
    }
    int rob(vector<int> &nums)
    {
        if (nums.empty())
            return 0;
        vector<int> dp = {0, nums[0]};
        for (int i = 1; i < nums.size(); i++)
        {
            int temp = max(dp[i], dp[i - 1] + nums[i]);
            dp.push_back(temp);
        }
        return dp[dp.size() - 1];
    }
};
int main()
{
    Solution mysolution;
    vector<int> temp = {1, 4, 1, 2, 1, 4};
    mysolution.rob(temp);
    return 0;
}