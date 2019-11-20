import sys
cost = 4
salary = 5
fee = 6
months = 5
pay = [11,9,10,14,9]

pay = [0]+pay
dp_matrix = [[sys.maxsize for month in range(months)] for _ in range(max(pay)+1)]
for employ in range(max(pay)+1):
    dp_matrix[employ][0] = employ*(cost+salary)
for month in range(1, months):
    can_list = range(pay[month], max(pay)+1)
    for employ in range(max(pay)+1):
        for can in can_list:
            dis = employ-can
            if dis > 0:
                temp_pay = employ*salary+dis*cost+dp_matrix[can][month-1]
            else:
                temp_pay = employ*salary-dis*fee+dp_matrix[can][month-1]
            dp_matrix[employ][month] = min(temp_pay, dp_matrix[employ][month])
pass
