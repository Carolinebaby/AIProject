# -*- coding: utf-8 -*-
# Author: Wu Yingfei
# Date: 2024-02-29
from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(edges):  # 无向图的可视化
    G = nx.Graph()

    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()


def build_graph(edges):  # 构造邻接链表
    idx_map = {}  # 顶点名称:顶点下标 键值对容器
    dot_map = []  # dot_map[i] 表示 下标i对应的顶点名称
    idx_map_size = 0
    graph = defaultdict(list)

    for x, y, w in edges:  # # 如果 dot_map 中没有 x 对应的点
        if x not in idx_map:
            idx_map[x] = idx_map_size
            dot_map.append(x)
            idx_map_size += 1
        if y not in idx_map:
            idx_map[y] = idx_map_size
            dot_map.append(y)
            idx_map_size += 1

        graph[idx_map[x]].append((idx_map[y], w))
        graph[idx_map[y]].append((idx_map[x], w))

    return idx_map, dot_map, idx_map_size, graph


def dijkstra(graph, start, end, idx_map_size):  # dijkstra 算法
    inf = 10 ** 9
    dist = [inf] * (idx_map_size + 1)
    mid_dot = [-1] * (idx_map_size + 1)

    dist[start] = 0
    minHeap = []
    heapq.heappush(minHeap, (0, start))

    while minHeap:
        dis, now = heapq.heappop(minHeap)
        for nextNode, dis in graph[now]:  # 遍历和当前点相邻的顶点
            if dist[now] + dis < dist[nextNode]:   # 比较新旧路径的长度，如果旧路径更长，则更新最短路径长度和中间点信息
                dist[nextNode] = dist[now] + dis
                if now != start:  # 如果当前点不是起点，则更新中间点信息
                    mid_dot[nextNode] = now
                heapq.heappush(minHeap, (dist[nextNode], nextNode))

    return dist[end], mid_dot


def floyd(edges, start, end, idx_map, idx_map_size):
    inf = 10 ** 9

    # 初始化二维数组 dist 和 mid_dot
    dist = [[inf] * idx_map_size for _ in range(idx_map_size)]
    mid_dot = [[-1] * idx_map_size for _ in range(idx_map_size)]

    # 初始化 dist，将已知边的权值赋给相应位置
    for x, y, w in edges:
        dist[idx_map[x]][idx_map[y]] = w
        dist[idx_map[y]][idx_map[x]] = w

    for i in range(idx_map_size):
        dist[i][i] = 0
        mid_dot[i][i] = i

    for i in range(idx_map_size):
        for j in range(idx_map_size):
            mid_dot[i][j] = i

    # Floyd 算法核心部分
    for k in range(idx_map_size):
        for i in range(idx_map_size):
            for j in range(idx_map_size):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    mid_dot[i][j] = mid_dot[k][j]

    return dist[start][end], mid_dot


def reconstruct_path(mid_dot, start, end, dot_map, choice):  # 根据中间点数组来重构最短路径
    if choice == 1:
        rpath = [dot_map[end]]
        nextNode = mid_dot[end]

        while nextNode != -1:
            rpath.append(dot_map[nextNode])
            nextNode = mid_dot[nextNode]

        rpath.append(dot_map[start])
        path = rpath[::-1]
        return path

    rpath = [dot_map[end]]
    nextNode = mid_dot[start][end]

    while nextNode != start:
        rpath.append(dot_map[nextNode])
        nextNode = mid_dot[start][nextNode]

    rpath.append(dot_map[start])
    path = rpath[::-1]
    return path


def main():  # 主函数
    m, n = map(int, input().split())  # 获取输入

    edges = [input().split() for _ in range(n)]
    edges = [(x, y, int(w)) for x, y, w in edges]  # 把路径长度信息修改成 int 类型
    idx_map, dot_map, idx_map_size, graph = build_graph(edges)  # 构造邻接链表
    visualize_graph(edges)  # 图的可视化

    start_str, end_str = input().split()  # 获取起点和终点的信息
    if start_str not in idx_map or end_str not in idx_map:  # 处理输入异常的情况
        print("no path")
    start = idx_map[start_str]  # 获取起点的下标信息
    end = idx_map[end_str]  # 获取终点的下标信息

    choice = int(input("使用哪种算法计算？ 1. dijkstra  2. floyd"))

    if choice == 1:
        # 实现 dijkstra 算法
        shortest_distance, mid_dot = dijkstra(graph, start, end, idx_map_size)
        # 根据中间点数组重构最短路径
        path = reconstruct_path(mid_dot, start, end, dot_map, 1)
        if shortest_distance == 10 ** 9:  # 起点和终点不连通
            print("no path")
        print(path)  # 输出最短路径
        print(shortest_distance)  # 输出最短路径长度
    else:
        shortest_distance, mid_dot = floyd(edges, start, end, idx_map, idx_map_size)
        path = reconstruct_path(mid_dot, start, end, dot_map,2)
        if shortest_distance == 10 ** 9:
            print("no path")
        print(path)
        print(shortest_distance)


if __name__ == "__main__":
    main()
