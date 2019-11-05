
import networkx as nx
'''
import numpy as np
from matplotlib import pyplot as plt
'''
import heapq
import sys


class Solution():
    def kuruste(self, graph):
        # 图的最小生成树算法
        node_num = len(graph)
        edge_list = []
        union_set = [-1 for _ in range(node_num)]
        for i in range(node_num):
            for j in range(i+1, node_num):
                if graph[i][j]:
                    edge_list.append([i, j, graph[i][j]])
        edge_list_sort = sorted(edge_list, key=lambda x: x[2])  # 初始化图需要通过边的权重排序

        def find_root(vertex, depth):  # 需要记录深度，以控制并查集的深度
            if union_set[vertex] == -1:
                return(vertex, depth)
            else:
                return(find_root(union_set[vertex], depth+1))
        result = 0
        for v0, v1, weight in edge_list_sort:
            v0_root, v0_depth = find_root(v0, 1)
            v1_root, v1_depth = find_root(v1, 1)
            if v0_root != v1_root:
                result += weight
                if v0_depth > v1_depth:
                    union_set[v1_root] = v0_root
                else:
                    union_set[v0_root] = v1_root
        return result

    def prime(self, graph):
        # 适用于稠密图
        def find_index(heapq_list, target):
            for index, item in enumerate(heapq_list):
                if item[0] == target:
                    return index
        temp_heapq = [[node, sys.maxsize]for node in range(len(graph))]
        temp_heapq[0][1] = 0  # 这个相当于存在一个哨兵结点，哨兵结点连接了结点0
        result = 0
        cutted_node = []
        while temp_heapq:
            # 每次取出最短的哪个元素，并做相应的计算
            choosed_node, choosed_length = temp_heapq.pop(0)
            result += choosed_length
            cutted_node.append(choosed_node)
            # 以最短元素为基础，更新heapq
            for neighbor, length in enumerate(graph[choosed_node]):  # 更新heapq
                if neighbor not in cutted_node:
                    nei_index = find_index(temp_heapq, neighbor)
                    if length and temp_heapq[nei_index][1] > length:
                        temp_heapq[nei_index][1] = length
                        while temp_heapq[nei_index][1] < temp_heapq[nei_index//2][1]:  # 上滤
                            temp_heapq[nei_index], temp_heapq[nei_index//2] = temp_heapq[nei_index//2], temp_heapq[nei_index]
                            nei_index = nei_index//2
        return result

    def find_match(self, left_num, right_num, graph):
        # 二分图匹配算法
        # 注意二分图可以以另一种方式存储
        left_target = [-1 for _ in range(left_num)]
        right_target = [-1 for _ in range(right_num)]
        right_visited = [False for _ in range(right_num)]
        result = 0

        def dfs(left_id):
            for right_id in range(left_num, left_num+right_num):
                pass
        for left_id in range(left_num):
            if left_target[left_id] == -1:
                right_visited = [False for _ in right_visited]
                result += dfs(left_id)
        return result

        '''     
        #include<bits/stdc++.h>
        #define MAXN 9999
        using namespace std;
        int nx,ny;//nx表示二分图左边顶点的个数，ny表示二分图右边顶点的个数
        int m;//m代表边的条数
        int cx[MAXN],cy[MAXN];//如果有cx[i]=j，则必有cy[j]=i，说明i点和j点能够匹配
        int x,y;//x点到y点有边
        int e[MAXN][MAXN];//邻接矩阵
        int visited[MAXN];//标记数组，标记的永远是二分图右边的顶点
        int ret;//最后结果
        int point(int u)//这个函数的作用是寻找增广路和更新cx，xy数组，如果找到了增广路，函数返回1，找不到，函数返回0。
        {
            for(int v=1;v<=ny;v++)//依次遍历右边的所有顶点
            {
                if(e[u][v]&&!visited[v])//条件一：左边的u顶点和右边的v顶点有连通边，条件二：右边的v顶点在没有被访问过，这两个条件必须同时满足
                {
                    visited[v]=1;//将v顶点标记为访问过的
                    if(cy[v]==-1||point(cy[v]))//条件一：右边的v顶点没有左边对应的匹配的点，条件二：以v顶点在左边的匹配点为起点能够找到一条增广路（如果能够到达条件二，说明v顶点在左边一定有对应的匹配点）。
                    {
                        cx[u]=v;//更新cx，cy数组
                        cy[v]=u;
                        return 1;
                    }
                }
            }
            return 0;//如果程序到达了这里，说明对右边所有的顶点都访问完了，没有满足条件的。
        }
        int main()
        {
            while (cin>>m>>nx>>ny)
            {
                memset(cx,-1,sizeof(cx));//初始化cx，cy数组的值为-1
                memset(cy,-1,sizeof(cy));
                memset(e,0,sizeof(e));//初始化邻接矩阵
                ret=0;
                while (m--)//输入边的信息和更新邻接矩阵
                {
                    cin>>x>>y;
                    e[x][y]=1;
                }
                for(int i=1;i<=nx;i++)//对二分图左边的所有顶点进行遍历
                {
                    if(cx[i]==-1)//如果左边的i顶点还没有匹配的点，就对i顶点进行匹配
                    {
                        memset(visited,0,sizeof(visited));//每次进行point时，都要对visited数组进行初始化
                        ret+=point(i);//point函数传入的参数永远是二分图左边的点
                    }
                }
                cout<<ret<<endl;
            }
        
        }
        '''


def list_to_graph(edge_list):
    node_num = max([node for edge in edge_list for node in edge])+1
    graph_matrix = [[0 for col in range(node_num)]for row in range(node_num)]
    for v0, v1 in edge_list:
        graph_matrix[v0][v1] = 1
        graph_matrix[v1][v0] = 1
    return graph_matrix


if __name__ == '__main__':
    solu = Solution()
    graph_matrix = [[0, 1, 7, 0, 0, 2, 3], [1, 0, 0, 9, 0, 1, 5], [7, 0, 0, 6, 8, 4, 0], [0, 9, 6, 0, 0, 0, 0], [0, 0, 8, 0, 0, 9, 0], [2, 1, 4, 0, 9, 0, 7], [3, 5, 0, 0, 0, 7, 0]]
    graph_dict = {(0, 1): 1, (0, 2): 7, (0, 5): 2, (0, 6): 3, (1, 3): 9, (1, 5): 1, (1, 6): 5, (2, 3): 6, (2, 4): 8, (2, 5): 4, (4, 5): 9, (5, 6): 7}
    bi_graph_list = [(0, 6), (0, 9), (0, 10), (1, 9), (1, 11), (2, 6), (2, 8), (3, 7), (4, 8), (4, 10), (5, 6), (5, 8)]
    # result = solu.prime(graph_matrix)
    result = solu.find_match(6, 6, list_to_graph(bi_graph_list))
