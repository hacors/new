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
	vector<vector<int>> threeSum(vector<int> &nums)
	{
		sort(nums.begin(), nums.end());
		//_printvector(nums);
		vector<vector<int>> result;
		for (int mid_index = 1; mid_index < nums.size() - 1; mid_index++)
		{
			int begin_index=0;
			int end_index=nums.size()-1;
			while(begin!=)
		}
		return result;
	}
};
int main()
{
	vector<int> temp = {-2, -1, 1, 2, 3};
	Solution mysolu;
	mysolu.threeSum(temp);
	return 0;
}