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
	int minRefuelStops(int target, int startFuel, vector<vector<int>> &stations)
	{
		stations.push_back({target, 0});
		stations.insert(stations.begin(), {0, startFuel});
		vector<vector<long long>> imfo = {4, vector<long long>(0)};
		long long tempsum = 0;
		for (int i = 0; i < stations.size(); i++)
		{
			tempsum += stations[i][1];
			imfo[0].push_back(stations[i][0]);
			imfo[1].push_back(stations[i][1]);
			imfo[2].push_back(tempsum);
		}
		for (int i = 0; i < stations.size() - 1; i++)
		{
			imfo[3].push_back(stations[i + 1][0]);
		}
		imfo[3].push_back(0);
		_printvectors(imfo);
		int count = stations.size() - 1;
		for (int i = 0; i < stations.size(); i++)
		{
			if (imfo[2][i] < imfo[3][i])
				return -1;
			if (imfo[2][i] - imfo[3][i] <= imfo[1][i])
				count = count - 1;
		}
		return (count);
	}
	int maxProfit(vector<int> &prices)
	{
		int buy_1 = -__INT_MAX__ - 1, buy_2 = -__INT_MAX__ - 1;
		int sell_1 = 0, sell_2 = 0;
		for (int i = 0; i < prices.size(); i++)
		{
			buy_1 = max(buy_1, -prices[i]);
			sell_1 = max(sell_1, prices[i] + buy_1);
			buy_2 = max(buy_2, sell_1 - prices[i]);
			sell_2 = max(sell_2, prices[i] + buy_2);
		}
		return sell_2;
	}
};
int main()
{
	Solution mysolu;
	vector<int> temp = {3, 3, 5, 0, 0, 3, 1, 4};
	int result = mysolu.maxProfit(temp);
	cout << result << 123;
	return 0;
}