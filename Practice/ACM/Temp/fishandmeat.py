
import numpy as np

meatball_num = 10
fishball_num = 15
bawl_num = 12
maxball_num = max(meatball_num, fishball_num)
temp_array = np.zeros((bawl_num, maxball_num), dtype=int)
for temp in range(bawl_num):
    temp_array[temp][0] = temp+1
for temp in range(maxball_num):
    temp_array[0][temp] = 1
for bowl in range(1, bawl_num):
    for ball in range(1, maxball_num):
        if bowl > ball:
            temp_array[bowl][ball] = temp_array[ball][ball]
        else:
            temp_array[bowl][ball] = temp_array[bowl-1][ball]+temp_array[bowl][ball - bowl]
result = 0
for temp in range(1, bawl_num):
    result = temp_array[temp][meatball_num-1]+temp_array[bawl_num-temp][fishball_num-1]+result
print(temp_array)
print(result)
'''
M=10
N=15
K=12
mn=max(M,N)
dp=[[0]*(K+1) for i in range(mn+1)]
dp[0][1]=0
for i in range(1,mn+1):
    dp[i][1]=1
for i in range(1,mn+1):
    for j in range(2,K+1):
        if i<j:
            dp[i][j]=0
        elif i==j:
            dp[i][i]=1
        else:
            dp[i][j]=(dp[i-1][j-1]+dp[i-j][j])%10000

res=0
for i in range(1,K):
    for j in range(1,K-i+1):
        res+=(dp[M][i]*dp[N][j])%10000
'''
print(res)