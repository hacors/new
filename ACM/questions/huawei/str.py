import re
charlist = str('a2{b4[c3(A)]}')
charlist = re.sub('[{]|[[]', '(', charlist)
charlist = re.sub('[}]|[]]', ')', charlist)
charlist = list(charlist)
poslist=[]
resultlist=[]
for index in range(len(charlist)):
    if charlist[index] == '(':
        poslist.append(index)
    elif charlist[index] == ')':
        begin=poslist.pop()
        repeat =int(charlist[begin-1])
        templist=charlist[begin+1:index]*(repeat-1)
    else:
        
print(a)
