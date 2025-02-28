from time import time
from heuristic_function import *
import heapq

# 全局常量
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # 目标状态
HEURISTIC_FUNCTIONS = [manhattan_dist, euclidean_dist, chebyshev_dist]  # 存放启发式函数
INPUT_STATES = [
    [1, 15, 7, 10, 9, 14, 4, 11, 8, 5, 0, 6, 13, 3, 2, 12],
    [1, 7, 8, 10, 6, 9, 15, 14, 13, 3, 0, 4, 11, 5, 12, 2],
    [5, 6, 4, 12, 11, 14, 9, 1, 0, 3, 8, 15, 10, 7, 2, 13],
    [14, 2, 8, 1, 7, 10, 4, 0, 6, 15, 11, 5, 9, 3, 13, 12],
    [14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15],
    [6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4],
    [11, 3, 1, 7, 4, 6, 8, 2, 15, 9, 10, 13, 14, 12, 5, 0],
    [0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3]
]  # 8 个测试案例


# -------------------------------- 函数部分 -------------------------------------
def state_str(state: [int]):
    return '\n'.join(' '.join(map(str, state[i:i+4])) for i in range(0, 16, 4))


def get_path_and_move_nums(father: [(int, int)], end_idx):
    """
    输出解决问题路径上所有的状态
    :param father: 存储父节点下标的列表
    :param end_idx: 最后一个状态的下标
    :return: path, move_nums
    """
    state_idx_path = []  # 存储路径上所有状态的下标
    move_nums = []  # type: [int]
    while end_idx != -1:  # 第0个状态的父节点的下标设置为 -1, 当达到 -1 时候说明到了 原状态的节点
        state_idx_path.append(end_idx)
        (end_idx, move_step) = father[end_idx]  # 得到当前状态的父节点的下标
        if end_idx != -1:
            move_nums.append(move_step)

    state_idx_path.reverse()
    move_nums.reverse()
    return state_idx_path, move_nums


def print_solve_states(states: [[int]], path: [int], end_step: int):
    # 输出解决问题的所有步骤
    for i in range(end_step + 1):
        print("------------ step", i, "-------------")
        print(state_str(states[path[i]]))


# -------------------------------- 实现 A* 算法的函数 -------------------------------
def solvable(state: [int]) -> bool:
    pos_0 = state.index(0)
    count = 0  # 逆序对数
    temp = state.copy()
    temp.remove(0)
    for i in range(15):
        for j in range(i+1, 15):
            if temp[i] > temp[j]:
                count += 1
    if (count + int(pos_0/4)) % 2:
        return True
    return False


# 根据启发式函数，计算当前状态和目标状态的距离
def state_dist(state: [int], hf):
    return sum(hf[state[i]][i] for i in range(16))


# 获取所有可能的下一个状态
def generate_next_state(state: [int]):
    """
    产生下一个状态
    :param state: 当前状态
    :return: 所有可能的下一个状态
    """
    pos = state.index(0)  # 找到 0 在 state 中的下标
    new_states = []  # 存放新状态
    i, j = divmod(pos, 4)  # 得到 0 映射在 4*4 矩阵中的横纵坐标
    for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
        if 0 <= ni < 4 and 0 <= nj < 4:
            new_state = state.copy()
            new_state[pos], new_state[ni * 4 + nj] = new_state[ni * 4 + nj], new_state[pos]
            new_states.append((new_state, new_state[pos]))
    return new_states


# A* 算法实现的主体
def a_star_search(states: [[int]], father: [(int, int)], hf: [[int]]):
    """
    A* 搜索算法
    :param states: 存储所有状态的数组
    :param father: 存储所有状态的父节点的下标
    :param hf: 启发式函数
    :return: 达到目标使用的步骤数 end_step 和 最后一个状态在 states 中的下标
    """
    open_list = []  # 使用列表作为堆
    original_dist = state_dist(states[0], hf)  # 初始状态的启发式距离
    heapq.heappush(open_list, (original_dist, original_dist, 0, 0))  # 将初始状态放入堆中
    closed_list = set()  # 使用集合来存储关闭列表中的状态
    while open_list:
        (now_cost, now_dist, now_step, now_idx) = heapq.heappop(open_list)  # 从堆中取出优先级最高的状态
        if now_dist == 0:
            return now_step, now_idx    # 如果当前状态已达到目标状态，则返回步骤数和状态在 states 中的下标

        closed_list.add(tuple(states[now_idx]))  # 将当前状态加入已处理集合中

        next_states = generate_next_state(states[now_idx])  # 生成当前状态的所有可能后继状态
        for (ns, move_num) in next_states:
            # if (tuple(ns) not in closed_list) and (solvable(ns)):  # 如果后继状态不在已处理集合中
            if tuple(ns) not in closed_list:  # 如果后继状态不在已处理集合中
                next_idx = len(states)  # 计算后继状态在 states 中的下标
                next_step = now_step + 1  # 更新后继状态的步数
                states.append(ns)  # 将后继状态加入 states 数组中
                father.append((now_idx, move_num))  # 更新后继状态的父节点下标
                next_dist = state_dist(ns, hf)  # 计算后继状态的启发式距离
                heapq.heappush(open_list, (next_dist + next_step, next_dist, next_step, next_idx))
                # 将后继状态加入堆中，元组格式为(总代价，启发式距离，步数，状态在 states 中的下标)


# 解决 15puzzle 的总函数
def solve_problem(original_state: [int], hf_kind, print_result_steps=False):
    """
    解决问题的总函数
    :param original_state: 初始状态
    :param hf_kind: 启发式函数的类型
    :param print_result_steps: 是否输出结果
    :return: None
    """
    if not solvable(original_state):
        print("无解")
        return

    father = list()  # type: [(int, int)]      # 存储父节点的下标，用于回溯输出
    states = list()  # type: [[int]]    # 存储所有状态

    # 初始化所有状态的列表和状态的父节点下标列表
    states.append(original_state)
    father.append((-1, -1))  # 初始状态的父节点设置为-1, 作为回溯得到路径上所有状态的时候的终止条件
    # 开始主体计算部分
    start_time = time()
    end_step, end_idx = a_star_search(states, father, HEURISTIC_FUNCTIONS[hf_kind])
    end_time = time()

    # 输出问题的解
    print("总时间: " + str(end_time - start_time) + "s")
    print("总步骤数:", end_step, "总拓展节点数:", len(states))
    path, move_nums = get_path_and_move_nums(father, end_idx)
    print("解题的序列:")
    move_str = ""
    for num in move_nums:
        move_str += str(num)+" "
    print(move_str)
    print("-----------------------------------")
    if print_result_steps:
        print("解题的步骤:")
        print_solve_states(states, path, end_step)  # 输出解决问题的所有步骤


# 主函数，可获取输入
def main():
    # 获取输入
    print("请选择使用的启发函数类型：1. 曼哈顿距离 2. 欧几里得距离 3. 切比雪夫距离")
    hf_kind = int(input())
    if hf_kind <= 0 or hf_kind >= 5:
        print("输入错误")
        return
    hf_kind -= 1
    print("请输入需要解决的 15puzzle")
    original_state = []
    for i in range(4):
        nums = [int(c) for c in input().strip().split(' ')]
        original_state += nums
    # 解决问题
    solve_problem(original_state, hf_kind, True)


# -----------------------------------------测试代码---------------------------------------------
def get_15puzzle_result(original_state, hf_kind, test_input_index):
    print("input", test_input_index)
    solve_problem(original_state, hf_kind)


def test_4_hard_input(hf_kind=0):
    for i in range(4):
        get_15puzzle_result(INPUT_STATES[i+4], hf_kind, i+5)


def test_4_simple_input(hf_kind=0):
    for i in range(4):
        get_15puzzle_result(INPUT_STATES[i], hf_kind, i + 1)


def test(hf_kind=0):
    print("-------------进行 8 个输入的测试-------------")
    print("examples 文件夹下的测试案例:")
    test_4_simple_input(hf_kind)
    print("ppt 中的四个测试案例:")
    test_4_hard_input(hf_kind)


def test_hf():
    print("---------启发式函数测试---------")
    print("启发式函数1: 曼哈顿距离")
    test_4_simple_input(0)
    print("***********************************")
    print("启发式函数2: 欧几里得距离")
    test_4_simple_input(1)
    print("***********************************")
    print("启发式函数3: 切比雪夫距离")
    test_4_simple_input(2)
    print("***********************************")


if __name__ == "__main__":
    main()
