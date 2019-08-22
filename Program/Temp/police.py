police = 15
thief = 17

list_of_rome = list()
for i in range(thief+1):
    list_of_rome.append(False)
list_of_route = list()
position = police
result = list()


def get_route(position, list_of_rome, list_of_route):
    if position != thief:
        if not list_of_rome[position+1]:
            list_of_rome[position] = True
            list_of_route.append(position)
            get_route(position+1, list_of_rome, list_of_route)
            list_of_route.pop()
            list_of_rome[position] = False
        if not list_of_rome[(position+int(thief/2)) % thief]:
            list_of_rome[position] = True
            list_of_route.append(position)
            get_route((position+int(thief/2)) % thief, list_of_rome, list_of_route)
            list_of_route.pop()
            list_of_rome[position] = False
        return
    else:
        print(list_of_route)
        result.append(list_of_route)


get_route(police, list_of_rome, list_of_route)
print(len(result))
