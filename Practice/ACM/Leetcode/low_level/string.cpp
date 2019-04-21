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
    string reverseString(string s)
    {
        string result;
        char next;
        result.clear();
        while (!s.empty())
        {
            next = s.back();
            s.pop_back();
            result.push_back(next);
        }
        return result;
    }
    int reverse(int x)
    {
        string nums;
        int result = 0;
        int reme;
        while (x)
        {
            reme = result;
            result = result * 10 + x % 10;
            x = x / 10;
        }
        if (result / 10 == reme)
            return result;
        else
            return 0;
    }
    int firstUniqChar(string s)
    {
        int store[26] = {0};
        for (int i = 0; i < s.length(); i++)
        {
            char tchar = s[i];
            int index = tchar - 'a';
            if (store[index] == 0)
                store[index] = i + 1;
            else
                store[index] = -1;
        }
        int small = s.length() + 1;
        for (int j = 0; j < 26; j++)
        {
            if (store[j] > 0)
            {
                if (store[j] < small)
                    small = store[j];
            }
        }
        if (small == s.length() + 1)
            return -1;
        else
            return small - 1;
    }
    bool isAnagram(string s, string t)
    {
        /*
        if (s.empty() && t.empty())
            return true;
        string s_ = s;
        string t_ = t;
        string::iterator iter_s, iter_t;
        for (iter_s = s.begin(); iter_s != s.end(); iter_s++)
        {
            for (iter_t = t.begin(); iter_t != t.end(); iter_t++)
            {
                if (*iter_s == *iter_t)
                {
                    s.erase(iter_s);
                    t.erase(iter_t);
                    if (s == t)
                        return true;
                    s = s_;
                    t = t_;
                }
            }
        }
        return false;
        */
        if (s.length() != t.length())
            return false;
        int count_s[26] = {0};
        int count_t[26] = {0};
        for (int i = 0; i < s.length(); i++)
        {
            count_s[s[i] - 'a'] += 1;
            count_t[t[i] - 'a'] += 1;
        }
        for (int j = 0; j < 26; j++)
        {
            if (count_s[j] != count_t[j])
                return false;
        }
        return true;
    }
    bool isPalindrome(string s)
    {
        //[48,57]+[65,90]+[97,122]
        string ori, reve;
        for (int i = 0; i < s.length(); i++)
        {
            if (s[i] <= 90 && s[i] >= 65)
            {
                ori.insert(ori.end(), s[i] + 32);
                reve.insert(reve.begin(), s[i] + 32);
            }
            else if ((s[i] <= 57 && s[i] >= 48) || (s[i] <= 122 && s[i] >= 97))
            {
                ori.insert(ori.end(), s[i]);
                reve.insert(reve.begin(), s[i]);
            }
        }
        if (ori == reve)
            return true;
        else
            return false;
    }
    int myAtoi(string str)
    {
        int symbol = 1;
        int pos = 0;
        int store = 0;
        int result = 0;
        while (pos < str.length() && str[pos] == ' ')
            pos++;
        if (pos < str.length() && str[pos] == '+')
            pos++;
        else if (pos < str.length() && str[pos] == '-')
        {
            pos++;
            symbol = -1;
        }
        while (pos < str.length() && _isnum(str[pos]))
        {
            store = result;
            result = result * 10 + str[pos] - '0';
            if (result / 10 != store)
                return symbol == 1 ? pow(2, 31) - 1 : -pow(2, 31);
            pos++;
        }
        return result * symbol;
    }
    bool _isnum(char judged)
    {
        if (judged <= 57 && judged >= 48)
            return true;
        else
            return false;
    }
    int strStr(string haystack, string needle)
    {
        if (needle.length() == 0)
            return 0;
        int *next = _buildnext(needle);
        int needlelen = needle.length();
        int haystacklen = haystack.length();
        int needlepos = 0;
        int haystackpos = 0;
        while (needlepos < needlelen && haystackpos < haystacklen)
        {
            if (needlepos < 0 || haystack[haystackpos] == needle[needlepos])
            {
                haystackpos++;
                needlepos++;
            }
            else
                needlepos = next[needlepos];
        }
        delete[] next;
        return (haystackpos == haystacklen && needlepos < needlelen) ? -1 : haystackpos - needlepos;
    }
    int *_buildnext(string needle)
    {
        int leng = needle.length();
        int *next = new int[leng];
        next[0] = -1;
        int m = 0;
        int t = -1;
        while (m < leng - 1)
        {
            if (t < 0 || needle[m] == needle[t])
            {
                m++;
                t++;
                next[m] = (needle[m] != needle[t]) ? t : next[t];
            }
            else
                t = next[t];
        }
        return next;
    }
    string countAndSay(int n)
    {
        string result = "1";
        string temp;
        int pos1, pos2;
        while (--n)
        {
            pos1 = 0;
            pos2 = 0;
            temp.clear();
            while (pos2 < result.length())
            {
                if (result[pos1] == result[pos2])
                    pos2++;
                else
                {
                    temp.insert(temp.end(), pos2 - pos1 + '0');
                    temp.insert(temp.end(), result[pos1]);
                    pos1 = pos2;
                }
            }
            temp.insert(temp.end(), pos2 - pos1 + '0');
            temp.insert(temp.end(), result[pos1]);
            result = temp;
        }
        return result;
    }
    string longestCommonPrefix(vector<string> &strs)
    {
        int leng = strs.size();
        string result;
        if (leng == 0)
            return result;
        int compare = 0;
        while (compare < strs[0].size())
        {
            char temp = strs[0][compare];
            for (int i = 0; i < leng; i++)
            {
                if (compare >= strs[i].size() || strs[i][compare] != temp)
                    return result;
            }
            result += temp;
            compare++;
        }
        return result;
    }
};
int main()
{
    Solution mysolution;
    cout << mysolution.countAndSay(4);
    return 0;
}