import copy
import time
from base_data import *
node_visited_count = 0  # 测试用的全局变量


def Search(board, is_black):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    color = BLACK if is_black else WHITE
    copy_board = copy.deepcopy(board)
    return AlphaBetaSearch(copy_board, color)


# --------------------------- 特殊情况判断 --------------------------
def CHONG_4_check(board, color):
    """
    敌方冲四棋型的判断
    :param board: 当前棋局
    :param color: 棋的颜色
    return: 如果没有冲四棋型返回-1,-1，如果有冲四棋型返回冲四棋型中的空位置
    """
    chong_4_patterns = CHONG_4_PATTERNS1[color]+CHONG_4_PATTERNS2[color]

    for pt in chong_4_patterns:
        locations = get_pattern_locations(board, pt)
        if not locations:
            continue
        x, y, direction = locations[0][0], locations[0][1], locations[0][2]
        for i in range(len(pt)):
            nx, ny = x + DIRE[direction][0] * i, y + DIRE[direction][1] * i
            if board[nx][ny] == EMPTY:
                return nx, ny
    return -1, -1


def HUO_4_check(board, color):
    locations = get_pattern_locations(board, HUO_4_PATTERN[color][0])
    if not locations:
        return False


# --------------------------- alpha-beta 剪枝的主函数 ------------------------------------
def AlphaBetaSearch(board, color):
    global node_visited_count
    node_visited_count = 0
    # 得到下一个可能下的位置
    next_move_locations = get_next_move_locations(board)
    # get_next_move_locations 有位置限定优化
    # 把可能的位置限定在一定范围内，不会把所有空位都放在 next_move_locations 里面

    # 预处理 1
    # 对于 冲四 的情况，直接处理
    x, y = CHONG_4_check(board, color)  # 检查我方棋子是否存在冲4
    if x != -1:
        return x, y
    x, y = CHONG_4_check(board, not color)  # 检查敌方棋子是否存在冲4
    if x != -1:
        return x, y

    # 预处理 2
    # 对于可以成为 活4 的情况，选择最优位置进行返回
    best_x, best_y, best_score = -1, -1, 0
    for x, y in next_move_locations:
        board[x][y] = color
        if get_pattern_count(board, HUO_4_PATTERN[color][0], color):  # 检查是否存在活4
            score = evaluation2(board)
            if score > best_score:
                best_x, best_y = x, y
        board[x][y] = EMPTY
    if best_x != -1:
        return best_x, best_y

    # alpha-beta 剪枝
    # 初始化 alpha, beta, end_x, end_y
    alpha = -1000000000000
    beta = 100000000000
    end_x, end_y = 0, 0
    # 根据 point_value 对 next_move_locations 进行排序
    next_move_locations.sort(key=lambda point: (point_value(point[0], point[1], board, color)), reverse=True)

    # 遍历可能的位置
    for x, y in next_move_locations:
        board[x][y] = color  # 修改当前位置的状态
        next_score = alpha_beta_search(alpha, beta, 1, not color, False, board)
        board[x][y] = EMPTY  # 恢复当前位置的状态
        if next_score > alpha:  # 更新 alpha, end_x, end_y
            end_x, end_y = x, y
            alpha = next_score
    print("访问状态数:", node_visited_count)
    return end_x, end_y


# ----------------------- alpha-beta 剪枝 功能实现的函数 -------------------------
def alpha_beta_search(alpha, beta, depth, color, is_max, board):
    global node_visited_count
    node_visited_count += 1
    # 判断是否到达叶子节点或者到达必胜状态
    if depth == MAX_DEPTH-1 or (get_pattern_count(board, CHENG_5_PATTERN[color][0], color) + get_pattern_count(board, HUO_4_PATTERN[color][0], color)>0):
        return evaluation2(board)

    # 得到棋子可能放置的位置
    next_move_locations = get_next_move_locations(board)
    # 如果遇到对方的冲四，直接作为下一步的下棋位置
    x, y = CHONG_4_check(board, not color)
    if x != -1:
        next_move_locations = [(x, y)]

    # 对 next_move_locations 进行排序
    next_move_locations.sort(key=lambda point: (point_value(point[0], point[1], board, color)), reverse=True)
    if is_max:  # max 层
        for x, y in next_move_locations:
            board[x][y] = color     # 修改 board[x][y] 的状态
            # 得到当前状态的启发函数值
            next_score = alpha_beta_search(alpha, beta, depth + 1, not color, False, board)
            board[x][y] = EMPTY     # 恢复 board[x][y] 的状态
            alpha = max(alpha, next_score)  # 更新 alpha 值
            if beta <= alpha:       # 剪枝
                break
        return alpha
    else:  # min 层
        for x, y in next_move_locations:
            board[x][y] = color
            next_score = alpha_beta_search(alpha, beta, depth + 1, not color, True, board)
            board[x][y] = EMPTY
            beta = min(beta, next_score)
            if beta <= alpha:
                break
        return beta


def evaluation(now_board):
    """
    判断当前棋局的分数
    """
    def evaluate(color, board):
        """
        判断棋局的分数
        :param color: 针对 color棋型计算棋局分数
        :param board: 当前棋局
        return: score 当前棋局的分数
        """
        score = 0  # 总的分数
        pattern_scores = [5000000, 4000000, 500, 400, 600, 550, 300, 200, 150, 70, 80, 20, 5]  # 不同棋型的分数
        #                  成5      活4       冲4  冲4   活3  活3  眠3  眠3  眠3  活2  活2  眠2 活1
        pattern_counts = []  # 不同棋型的统计数量
        p_size = len(pattern_scores)
        # 遍历所有的棋型
        for i in range(p_size):
            cnt = 0
            if i == 4:  # 连活三, 需要特殊判断
                for pt in PATTERNS[color][i]:
                    cnt += get_pattern_count(board, pt, color, is_HUO_3=True)
            elif i == 9:  # 连活二, 也需要特殊判断
                for pt in PATTERNS[color][i]:
                    cnt += get_pattern_count(board, pt, color, is_HUO_2=True)
            else:
                for pt in PATTERNS[color][i]:
                    cnt += get_pattern_count(board, pt, color)
            pattern_counts.append(cnt)

        # 计算总分
        for i in range(p_size):
            score += pattern_scores[i] * pattern_counts[i]

        # 特殊棋型
        if pattern_counts[2] + pattern_counts[3] >= 2:  # 双冲四
            score += 800000
        elif pattern_counts[2] + pattern_counts[3] >= 1 and pattern_counts[4] + pattern_counts[5] >= 1:  # 冲四+冲三
            score += 700000
        elif pattern_counts[4] + pattern_counts[5] >= 2:  # 双冲三
            score += 600000
        return score
    # 返回总的评估值
    return evaluate(BLACK, now_board)-evaluate(WHITE, now_board)


def point_value(pos_x, pos_y, now_board, now_color):
    def static_table_value(point_x, point_y, dire, board, color, is_black):
        """
        计算单个方向的静态表值
        """
        own = 1 if color == is_black else 0  # 是否是 我方
        opp_color = not color

        start_x, start_y = point_x - 4 * dire[0], point_y - 4 * dire[1]  # 通过当前位置的五连通网格点的起点

        while not is_in_board(start_x, start_y):
            start_x, start_y = start_x + dire[0], start_y + dire[1]

        valid_pos = []  # 提前存好有效位置
        while is_in_board(start_x, start_y) and (start_x != point_x + 5 * dire[0] or start_y != point_y + 5 * dire[1]):
            valid_pos.append([start_x, start_y])
            start_x, start_y = start_x + dire[0], start_y + dire[1]
        # 如果有效位置长度小于5
        if len(valid_pos) < 5:
            return 0

        size, color_cnt, opp_cnt, value = 0, 0, 0, 0  # color_cnt我方棋子数量, opp_cnt 敌方棋子数量
        last_end_x, last_end_y = valid_pos[0][0], valid_pos[0][1]  # 五连通棋子的前面的一个棋子位置
        # 计算每个经过目标位置的五连通棋子的分数，并累加
        for x, y in valid_pos:
            if size < 5:
                size += 1
                color_cnt += 1 if board[x][y] == color else 0
                opp_cnt += 1 if board[x][y] == opp_color else 0
                if size == 5 and opp_cnt == 0:
                    value = STATIC_TABLE[own][color_cnt]
            else:
                color_cnt += (board[x][y] == color) - (board[last_end_x][last_end_y] == color)
                opp_cnt += (board[x][y] == opp_color) - (board[last_end_x][last_end_y] == opp_color)
                value += STATIC_TABLE[own][color_cnt] if not opp_cnt else 0
                last_end_x, last_end_y = last_end_x + dire[0], last_end_y + dire[1]

        return value
    # 计算当前位置的分数: max(own_score, opp_score)
    res = max(sum(static_table_value(pos_x, pos_y, d, now_board, now_color, now_color == BLACK) for d in DIRE),
              sum(static_table_value(pos_x, pos_y, d, now_board, not now_color, now_color == BLACK) for d in DIRE))
    return res


def get_next_move_locations(board):
    next_move_locations = []
    left, right, top, bottom = -1, -1, -1, -1
    for x in range(15):
        for y in range(15):
            if board[x][y] != EMPTY:
                if left == -1:
                    left, right, top, bottom = x, x, y, y
                else:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
    left = max(0, left - 2)
    right = min(14, right + 2)
    top = max(0, top - 2)
    bottom = min(14, bottom + 2)

    for x in range(left, right + 1):
        for y in range(top, bottom + 1):
            if board[x][y] == EMPTY:
                next_move_locations.append((x, y))
    return next_move_locations


def _check_pattern(board, x, y, pattern, dx, dy):
    for goal in pattern[1:]:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= 15 or y >= 15 or board[x][y] != goal:
            return False
    return True


def is_in_board(x: int, y: int):
    return 0 <= x < 15 and 0 <= y < 15


def get_pattern_count(board, pattern, color, is_HUO_3=False, is_HUO_2=False):
    count = 0

    for x in range(15):
        for y in range(15):
            if pattern[0] == board[x][y]:
                for dire_flag, dire in enumerate(DIRE):
                    if _check_pattern(board, x, y, pattern, dire[0], dire[1]):
                        if is_HUO_3:
                            x1, y1 = x-dire[0], y-dire[1]
                            x2, y2 = x+dire[0]*5, y+dire[1]*5
                            if ((is_in_board(x1, y1) and board[x1][y1] == EMPTY) or
                                    (is_in_board(x2, y2) and board[x2][y2] == EMPTY)):
                                if ((is_in_board(x2, y2) and board[x2][y2] == color) or
                                        (is_in_board(x1, y1) and board[x1][y1] == color)):
                                    count -= 1
                                count += 1
                        elif is_HUO_2:
                            x1, y1 = x-dire[0], y-dire[1]
                            x2, y2 = x+dire[0]*4, y+dire[1]*4
                            if ((is_in_board(x1, y1) and board[x1][y1] == EMPTY) or
                                    (is_in_board(x2, y2) and board[x2][y2] == EMPTY)):
                                count += 1
                        else:
                            count += 1
    return count


def get_pattern_locations(board, pattern):
    pattern_list = []
    for x in range(15):
        for y in range(15):
            if pattern[0] == board[x][y]:
                if len(pattern) == 1:
                    pattern_list.append((x, y, 0))
                else:
                    for dire_flag, dire in enumerate(DIRE):
                        if _check_pattern(board, x, y, pattern, dire[0], dire[1]):
                            pattern_list.append((x, y, dire_flag))
    return pattern_list


def evaluation2(now_board):
    def static_table_value(point_x, point_y, dire, board, color):
        """
        计算单个方向的静态表值
        """
        opp_color = not color

        start_x, start_y = point_x - 4 * dire[0], point_y - 4 * dire[1]  # 通过当前位置的五连通网格点的起点

        while not is_in_board(start_x, start_y):
            start_x, start_y = start_x + dire[0], start_y + dire[1]

        valid_pos = []  # 提前存好有效位置
        while is_in_board(start_x, start_y) and (start_x != point_x + 5 * dire[0] or start_y != point_y + 5 * dire[1]):
            valid_pos.append([start_x, start_y])
            start_x, start_y = start_x + dire[0], start_y + dire[1]
        # 如果有效位置长度小于5
        if len(valid_pos) < 5:
            return 0

        size, color_cnt, opp_cnt, value = 0, 0, 0, 0  # color_cnt我方棋子数量, opp_cnt 敌方棋子数量
        last_end_x, last_end_y = valid_pos[0][0], valid_pos[0][1]  # 五连通棋子的前面的一个棋子位置
        # 计算每个经过目标位置的五连通棋子的分数，并累加
        for x, y in valid_pos:
            if size < 5:
                size += 1
                color_cnt += 1 if board[x][y] == color else 0
                opp_cnt += 1 if board[x][y] == opp_color else 0
                if size == 5 and opp_cnt == 0:
                    value = STATIC_TABLE2[color_cnt]
            else:
                color_cnt += (board[x][y] == color) - (board[last_end_x][last_end_y] == color)
                opp_cnt += (board[x][y] == opp_color) - (board[last_end_x][last_end_y] == opp_color)
                value += STATIC_TABLE2[color_cnt] if not opp_cnt else 0
                last_end_x, last_end_y = last_end_x + dire[0], last_end_y + dire[1]

        return value

    def evaluate(board, color):
        value = 0
        for x in range(15):
            for y in range(15):
                if board[x][y] != EMPTY:
                    value += sum(static_table_value(x, y, d, board, color) for d in DIRE)
        return value
    return evaluate(now_board, BLACK)-evaluate(now_board, WHITE)


# --------------------------------------- 测试代码 ---------------------------------------
def test1():
    def print_matrix(matrix):
        for row in matrix:
            for i in range(len(row)):
                row[i] = str(row[i]).rjust(10)

        # 打印结果
        for row in matrix:
            print(' '.join(row))

    board = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 1, 0, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, 1, 0, 0, 1, -1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
    pos_values = []
    pos_evaluation = []
    for x in range(15):
        single_row = []
        for y in range(15):
            if board[x][y] == EMPTY:
                board[x][y] = BLACK
                single_row.append(evaluation2(board))
                board[x][y] = EMPTY
            else:
                single_row.append(0)
        pos_evaluation.append(single_row)

    print("---------------------------------评估函数值---------------------------------")
    print("pos_evaluation = ")
    print_matrix(pos_evaluation)

    # ------------------ 位置的静态表值 -------------------
    for i in range(15):
        single_row_pos_values = []
        for j in range(15):
            if board[i][j] == EMPTY:
                score = point_value(i, j, board, BLACK)
                single_row_pos_values.append(score)
            else:
                single_row_pos_values.append(0)
        pos_values.append(single_row_pos_values)
    print("--------------------------------位置的静态表值--------------------------------")
    print("pos_values = ")
    print_matrix(pos_values)
    print("----------------------------------alpha值---------------------------------")
    pos_alpha_values = []
    print("alpha值 = ")
    for i in range(15):
        single_row_alpha_values = []
        for j in range(15):
            if board[i][j] == EMPTY:
                board[i][j] = BLACK
                alpha = alpha_beta_search(-100000000000, 100000000000, 1, WHITE, False, board)
                board[i][j] = EMPTY
                single_row_alpha_values.append(alpha)
            else:
                single_row_alpha_values.append(0)
        pos_alpha_values.append(single_row_alpha_values)
    print_matrix(pos_alpha_values)

    return 0


def test2():
    board = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 1, 0, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, 1, 0, 0, 1, -1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
    x, y = AlphaBetaSearch(board, BLACK)
    print(x, y)


if __name__ == '__main__':
    test2()
