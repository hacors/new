N, M = list(map(int, input().split()))
m = []
for i in range(N):
    m.append(list(map(int, input().split())))
X, Y, Z, W = list(map(int, input().split()))
visit = [[0]*M for i in range(N)]


def find(x, y):
    if x == Z and y == W:
        return 1
    res = 0
    if x > 0 and m[x-1][y] > m[x][y]:
        res += find(x-1, y)
    if x < N-1 and m[x+1][y] > m[x][y]:
        res += find(x+1, y)
    if y > 0 and m[x][y-1] > m[x][y]:
        res += find(x, y-1)
    if y < M-1 and m[x][y+1] > m[x][y]:
        res += find(x, y+1)
    return res % 1000000000


print(find(X, Y))
