#include <vector>
#include <map>
#include <set>
#include <math.h>
#include <algorithm>
#include <iostream>
using namespace std;
void _printvector(vector<int> &nums)
{
    for (int i = 0; i < nums.size(); i++)
        cout << "  =>  " << nums[i];
    cout << endl;
}
void _printmap(map<char, int> &themap)
{
    map<char, int>::iterator iter;
    for (iter = themap.begin(); iter != themap.end(); iter++)
    {
        cout << iter->first << ':' << iter->second << ' ';
    }
    cout << endl;
}
void _printstring(string &thestring)
{
    cout << thestring << endl;
}