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
		cout << count;
	}
};
int main()
{
	Solution mysolu;
	vector<vector<int>> temp = {{10, 16}, {20, 30}, {30, 30}, {60, 40}};
	mysolu.minRefuelStops(100, 10, temp);
	return 0;
}