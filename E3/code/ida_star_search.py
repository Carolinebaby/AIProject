from time import time
from heuristic_function import *

GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
HEURISTIC_FUNCTIONS = [manhattan_dist, euclidean_dist, chebyshev_dist]         # 存放启发式函数
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


# 根据启发式函数，计算当前状态和目标状态的距离
def state_dist(state: [int], hf):
    return sum(hf[state[i]][i] for i in range(16))


def state_str(state: [int]):
    return '\n'.join(' '.join(map(str, state[i:i+4])) for i in range(0, 16, 4))


# 判断是否可解
def solvable(state: [int]) -> bool:
    pos_0 = state.index(0)
    count = 0
    for i in range(16):
        for j in range(i+1, 16):
            if state[i] > state[j]:
                count += 1
    if (count + pos_0/4) % 2:
        return True
    return False


# IDA* 搜索
def ida_star_search(original_state: [int], hf):
    """
    IDA* 搜索
    :param original_state: 初始状态
    :param hf: 启发式函数
    :return: stack, 终止的步数step, end_move_num
    """
    upper_bound = state_dist(original_state, hf)  # 初始化上界为原始状态到目标状态的估计距离
    stack = []  # 初始化栈

    while 1:  # 无限循环直到找到解
        # 将原始状态和相关信息压入栈 dist, state, pos_0, step, move_num
        stack.append([state_dist(original_state, hf), original_state, original_state.index(0), 0, -1])
        next_states = []  # 初始化下一个可能状态的列表
        visited = set()  # 初始化已访问状态的集合

        while stack:  # 当栈不为空时
            [s_dist, state, s_pos0, step, mn] = stack[-1]  # 获取栈顶元素

            if tuple(state) in visited:  # 如果状态已访问
                stack.pop()  # 弹出栈顶元素
                visited.remove(tuple(state))  # 从已访问集合中移除该状态
                continue
            visited.add(tuple(state))  # 将新状态添加到已访问集合

            if s_dist == 0:  # 如果当前状态到目标状态的距离为0
                return stack, step, mn  # 返回路径和步数

            i, j = divmod(s_pos0, 4)  # 计算0的位置
            # 遍历0的上下左右的位置
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < 4 and 0 <= nj < 4:  # 如果新位置在边界内
                    new_state = state.copy()  # 复制当前状态
                    # 交换0和新位置的值
                    new_state[s_pos0], new_state[ni * 4 + nj] = new_state[ni * 4 + nj], new_state[s_pos0]
                    if tuple(new_state) not in visited:  # 如果新状态未访问
                        next_dist = state_dist(new_state, hf)  # 计算新状态到目标状态的距离
                        # 如果新状态的距离加上步数小于等于上界
                        if next_dist + step + 1 <= upper_bound:
                            next_states.append([next_dist, new_state, ni*4+nj, step+1, new_state[s_pos0]])  # 添加到下一个可能状态列表

            next_states.sort()  # 对可能的下一个状态进行排序
            while next_states:  # 当有可能的下一个状态时
                stack.append(next_states[-1])  # 将最小距离的状态压入栈
                next_states.pop()  # 移除已经压入栈的状态
        upper_bound += 2  # 增加上界


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

    hf = HEURISTIC_FUNCTIONS[hf_kind]
    # 开始主体计算部分
    start_time = time()
    stack, end_step, end_move = ida_star_search(original_state, hf)
    end_time = time()
    # 输出结果
    print("总时间:", str(end_time - start_time) + "s")
    print("总的步骤数:", end_step)
    now_output = 0
    last = stack[0]
    print("解题的序列:")
    move_str = ""
    for s in stack:
        steps = s[3]
        if steps > now_output:
            if last[4] != -1:
                move_str += str(last[4])+" "
            now_output += 1
        last = s
    move_str += str(end_move)
    print(move_str)
    print("---------------------------------")
    now_output = 0
    last = stack[0]
    if print_result_steps:
        print("解题的步骤:")
        for s in stack:
            steps = s[3]
            if steps > now_output:
                x = last[1]
                print("------------ step", now_output, "-------------")
                now_output += 1
                print(state_str(x))
            last = s
        print("------------ step", end_step, "-------------")  # stack 中没有保存目标状态，所以需要单独输出
        print(state_str(GOAL_STATE))


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


# -----------------------------测试代码-------------------------------
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
