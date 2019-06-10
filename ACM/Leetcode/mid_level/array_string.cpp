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
		for (int begin_index = 0; begin_index < nums.size() - 1; begin_index++)
		{
			int mid_index = begin_index + 1;
			int end_index = nums.size() - 1;
			while (mid_index != begin_index)
			{
				int temp_sum = nums[begin_index] + nums[mid_index] + nums[end_index];
				if (temp_sum > 0)
					end_index--;
				else if (temp_sum < 0)
					begin_index++;
				else
				{
					vector<int> temp_result = {nums[begin_index], nums[mid_index], nums[end_index]};
					_printvector(temp_result);
					mid_index++;
					end_index--;
					result.push_back(temp_result);
				}
			}
		}
		return result;
	}
};
int main()
{
	vector<int> temp = {-1, 0, 1, 2, -1, -4}; //{-2, -1, 1, 2, 3};
	Solution mysolu;
	vector<vector<int>> result = mysolu.threeSum(temp);
	return 0;
}