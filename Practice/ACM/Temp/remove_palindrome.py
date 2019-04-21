# include<cstdio>
# include<cstring>
inline int min(int a, int b) {return a < b?a: b
                              }
const int maxn = 20, inf = 0x3f3f3f3f
int dp[1 << maxn], len
bool ok[1 << maxn]
char s[maxn]
inline bool judge(int i) {
    int h = len, l = 0
    while(h > l) {
        while(!((1 << h) & i)) - -h
        while(!((1 << l) & i)) + +l
        if(s[h] != s[l]) return false
        - -h, ++l
    }
    return true
}
int main() {
    scanf("%s", s)
    len = strlen(s)
    memset(dp, inf, sizeof(dp))
    dp[0] = 0
    for(int i=1
        i < (1 << len)
        + +i) {
        if(judge(i)) {dp[i] = 1
                      continue
                      }
        for(int j=i
            j
            j=(j-1) & i) {
            if(j == i) continue
            dp[i] = min(dp[i], dp[i-j]+dp[j])
        }
    }
    printf("%d", dp[(1 << len)-1])
    return 0
}
