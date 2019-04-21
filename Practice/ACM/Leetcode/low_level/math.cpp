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
    vector<string> fizzBuzz(int n)
    {
        vector<string> result;
        for (int i = 1; i <= n; i++)
        {
            if (i % 15 == 0)
                result.push_back("FizzBuzz");
            else if (i % 3 == 0)
                result.push_back("Fizz");
            else if (i % 5 == 0)
                result.push_back("Buzz");
            else
                result.push_back(to_string(i));
        }
        return result;
    }
    int countPrimes(int n)
    {
        int res = 0;
        vector<bool> check(n, true);
        for (int i = 2; i <= n / 2; i++)
        {
            if (check[i])
            {
                res++;
                for (int j = 2; j * i < n; j++)
                {
                    check[j * i] = false;
                }
            }
        }
        return res;
    }
    bool isPowerOfThree(int n)
    {
        if (n == 0)
            return false;
        else if (n == 1)
            return true;
        else
            return n % 3 == 0 && isPowerOfThree(n / 3);
    }
    int romanToInt(string s)
    {
        map<char, int> match{{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}};
        int res = 0;
        for (int i = 0; i < s.length(); i++)
        {
            if (i + 1 == s.length() || match.at(s[i]) >= match.at(s[i + 1]))
                res += match.at(s[i]);
            else
                res -= match.at(s[i]);
        }
        return (res);
    }
};
int main()
{
    Solution mysolu;
    string temp = "IVX";
    mysolu.romanToInt(temp);
    return 0;
}