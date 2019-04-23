number = 9


def getchock(n):
    if n < 6:
        return 0
    else:
        templist = [1]
        for left_candy in range(1, n-5):
            temp = sum(templist)
            templist.append(temp)
        return sum(templist)

a = getchock(number)
print(a)
