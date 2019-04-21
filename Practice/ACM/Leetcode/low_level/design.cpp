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
    Solution(vector<int> nums)
    {
        store = outputs = nums;
    }
    vector<int> reset() { return store; }
    vector<int> shuffle() //洗牌算法
    {
        int n = outputs.size();
        for (int i = 0; i < n; i++)
        {
            int choosed = rand() % (n - i);
            swap(outputs[i], outputs[i + choosed]);
        }
        return outputs;
    }

  private:
    vector<int> store;
    vector<int> outputs;
};
class MinStack
{
  public:
    MinStack() { mins.push(__INT_MAX__); }
    void push(int x)
    {
        store.push(x);
        if (x <= mins.top())
            mins.push(x);
    }
    void pop()
    {
        int temp = store.top();
        store.pop();
        if (temp == mins.top())
            mins.pop();
    }
    int top()
    {
        return store.top();
    }
    int getMin()
    {
        return mins.top();
    }

  private:
    stack<int> store, mins;
};
int main()
{
    vector<int> temp = {1, 4, 1, 2, 1, 4};
    return 0;
}