#include <stdio.h>
#include <time.h>
#include <windows.h>
const int size = 512;
void init(long array[size][size])
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            array[i][j] = i * 10 + j;
        }
    }
}
void swap(long array[size][size])
{
    for (int i = 0; i < size; i++)
    {
        for (int j = i; j < size; j++)
        {
            long temp = array[i][j];
            array[i][j] = array[j][i];
            array[j][i] = temp;
        }
    }
}
int main()
{

    DWORD start, end;
    long origin_array[size][size];
    init(origin_array);
    start = GetTickCount();
    swap(origin_array);
    end = GetTickCount();
    printf("time=%d\n", end - start);
    return 0;
}