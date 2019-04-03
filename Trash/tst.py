def cango(N):
    cangolist = list()
    for i in range(1, N+1):
        cangolist.append([])
        for j in range(1, N+1):
            if ((not i == j) and (j % i == 0 or i % j == 0)):
                cangolist[i-1].append(j)
    return cangolist


solution = 0


def getroute(remain, pos, target, cango):
    global solution
    if not remain == 0:
        if(pos == target):
            solution += 1
            return
        else:
            for choose in cango[pos-1]:
                getroute(remain-1, choose, target, cango)
    else:
        return


class miancode():
    def maxcircles(self, input1, input2, input3):
        global solution
        solution = 0
        cangolist = cango(input1)
        getroute(input3, input2, input2, cangolist)
        print(solution)
        return solution


my = miancode()
my.maxcircles(3, 2, 4)
