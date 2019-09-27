'''
import networkx as nx
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


if __name__ == '__main__':
    solu = Solution()
    graph_matrix = [[0, 1, 7, 0, 0, 2, 3], [1, 0, 0, 9, 0, 1, 5], [7, 0, 0, 6, 8, 4, 0], [0, 9, 6, 0, 0, 0, 0], [0, 0, 8, 0, 0, 9, 0], [2, 1, 4, 0, 9, 0, 7], [3, 5, 0, 0, 0, 7, 0]]
    graph_list = {(0, 1): 1, (0, 2): 7, (0, 5): 2, (0, 6): 3, (1, 3): 9, (1, 5): 1, (1, 6): 5, (2, 3): 6, (2, 4): 8, (2, 5): 4, (4, 5): 9, (5, 6): 7}
    result = solu.prime(graph_matrix)
    '''
    solu.prime(graph)
    graph_np = np.array(graph)
    netw = nx.from_numpy_matrix(graph_np)
    nx.draw(netw, with_labels=True, label='weight')
    plt.show()
    '''
    pass
