import copy

pos_in_set = ['a', 'b', 'c', 'd', 'e', 'f']


class Word:
    def __init__(self, word_str):
        self.arguments = []  # 字符串列表，每个字符串代表一个参数
        self.is_negative = False  # 谓词前面是否有 ¬
        self.predicate = ""  # 谓词
        pos = 0
        if word_str[0] == '¬':  # 如果谓词前面有 ¬
            self.is_negative = True
            pos = 1
        temp = ""
        # 获取谓词和参数
        while pos < len(word_str):
            if word_str[pos] == "(":  # 如果遇到(，说明前面的是谓词
                self.predicate = temp
                temp = ""
                pos += 1
                continue
            # 如果遇到 , 或 ) 说明前面的是 argument
            if word_str[pos] == "," or word_str[pos] == ')':
                self.arguments.append(temp)
                temp = ""
                pos += 1
                continue
            temp += word_str[pos]
            pos += 1

    def __str__(self):
        arguments_str = ",".join(self.arguments)
        negative_str = "¬" if self.is_negative else ""
        return f"{negative_str}{self.predicate}({arguments_str})"

    # 替换变量
    def replace_var(self, old_name: [str], new_name: [str]):
        for i in range(len(old_name)):
            self.arguments = [new_name[i] if arg == old_name[i] else arg for arg in self.arguments]

    def __lt__(self, other):  # sort 子句中的文字的时候会用到
        return self.predicate < other.predicate


class Step:
    def __init__(self, pos1, pos2, old_name: [], new_name: [], clause: [Word]):
        self.pos1 = pos1  # 第一个文字的位置
        self.pos2 = pos2  # 第二个文字的位置
        self.old_name = copy.deepcopy(old_name)  # 被替换的变量
        self.new_name = copy.deepcopy(new_name)	 # 替换的新名称
        self.clause = clause  # 归结生成的子句

    def __str__(self):
        replace_str = ""
        if self.old_name:
            replace_str += "{" + ','.join(f"{old}={new}" for old, new in zip(self.old_name, self.new_name)) + "}"
        c_str = "[]" if not len(self.clause) else get_clause_str(self.clause)
        return f"R[{self.pos1},{self.pos2}]{replace_str} = {c_str}"


def is_same(w1: Word, w2: Word) -> bool and int:
    for i in range(len(w1.arguments)):
        if w1.arguments[i] != w2.arguments[i]:
            return False, i
    return True, 0


def is_var(var: str) -> bool:
    return var in ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# 前提: 两个文字谓词相同，互为否定，参数不同
def change(pos: int, w1: Word, w2: Word):
    # pos: 第一个不同的参数的位置
    # 如果当前位置的参数都是常量，则两个文字不能进行归结
    if not is_var(w1.arguments[pos]) and not is_var(w2.arguments[pos]):
        return False, "", ""

    if is_var(w1.arguments[pos]):  # 第一个文字的参数是变量
        old_name = w1.arguments[pos]
        new_name = w2.arguments[pos]
        w1.replace_var([old_name], [new_name])
        return True, old_name, new_name
    else:  # 第二个文字的参数是变量
        old_name = w2.arguments[pos]
        new_name = w1.arguments[pos]
        w2.replace_var([old_name], [new_name])
        return True, old_name, new_name


def judge(w1: Word, w2: Word):
    word1 = copy.deepcopy(w1)  # 深拷贝，防止修改原来的文字
    word2 = copy.deepcopy(w2)
    old_name = []
    new_name = []
    # 不能归结的情况：谓词不同，参数数目不同，不是互为否定
    if word1.predicate != word2.predicate or len(word1.arguments) != len(word2.arguments) or (
            word1.is_negative and word2.is_negative) or (not word1.is_negative and not word2.is_negative):
        return False, old_name, new_name

    # flag: 是否能够归结, dif_pos 通过 is_same 函数返回第一个不同地方（根据前面排除的情况可以知道这个地方一定是第一个不同的参数的位置）
    flag, dif_pos = is_same(word1, word2)

    while not flag:  # 如果不相同
        is_changed, old, new = change(dif_pos, word1, word2)
        if not is_changed:  # 两个不同的参数都是常量，则无法归结
            return False, old_name, new_name
        old_name.append(old)  # 把变量替换方案放入总的替换方案中
        new_name.append(new)
        flag, dif_pos = is_same(word1, word2)  # 继续判断

    return True, old_name, new_name


def get_clause_str(clause: [Word]) -> str:
    clause_str = ""
    size = len(clause)
    for i in range(size):
        clause_str += str(clause[i])
        if i != size-1:
            clause_str += ","
    return clause_str


def union(clause1: [Word], clause2: [Word], pos1: [], pos2: [], old_name: [str], new_name: [str]) -> [Word]:
    c1 = copy.deepcopy(clause1)  # 深拷贝，防止修改原来的Word对象
    c2 = copy.deepcopy(clause2)
    # 变量替换
    for i in range(len(c1)):
        c1[i].replace_var(old_name, new_name)

    for i in range(len(c2)):
        c2[i].replace_var(old_name, new_name)

    uc = []  # 两个子句归结后的子句
    for i in range(len(c1)):
        if i != pos1:  # 不是被归结位置上的文字
            new_flag = True
            for word in uc:
                if str(c1[i]) == str(word):
                    new_flag = False
            if new_flag:  # 如果在子句中不存在
                uc.append(c1[i])

    for i in range(len(c2)):
        if i != pos2:
            new_flag = True
            for word in uc:
                if str(c2[i]) == str(word):
                    new_flag = False
            if new_flag:
                uc.append(c2[i])

    return uc


def is_exist(clause: [Word], clause_set: [[Word]]) -> bool:
    for cs in clause_set:
        if get_clause_str(clause) == get_clause_str(cs):
            return True
    return False


def unity(steps: [], clause_set: [[Word]], c1_idx, c2_idx):
    c1 = copy.deepcopy(clause_set[c1_idx])
    c2 = copy.deepcopy(clause_set[c2_idx])
    end_flag = False
    end_step_idx = -1

    for i, w1 in enumerate(c1):
        for j, w2 in enumerate(c2):
            flag, old_name, new_name = judge(w1, w2)
            if flag:
                uc = union(c1, c2, i, j, old_name, new_name)
                uc.sort()

                if not uc:  # 如果归结后的子句为空
                    end_flag = True
                    end_step_idx = len(steps) + 1

                if is_exist(uc, clause_set):  # 避免子句重复
                    continue
                # 如果没有重复，则把 归结后的子句存入 clause_set 中
                clause_set.append(uc)

                # 得到 当前步骤的信息 然后创建 Step对象，保存在 steps 列表中
                pos1_str = f"{c1_idx + 1}{pos_in_set[i]}" if len(c1) > 1 else f"{c1_idx + 1}"
                pos2_str = f"{c2_idx + 1}{pos_in_set[j]}" if len(c2) > 1 else f"{c2_idx + 1}"


                steps.append(Step(pos1_str, pos2_str, old_name, new_name, uc))

    return end_flag, end_step_idx


def get_pos(p_str: str) -> int:
    temp = copy.deepcopy(p_str)
    str_len = len(temp)
    if temp[str_len-1] in pos_in_set:
        temp = temp[:str_len-1]
    return int(temp)


def back_search(steps: [Step], end_step_idx: int):
    step_list = []
    q = [end_step_idx]
    while q:
        top = q.pop(0)
        top -= 1
        step_list.append(top)
        step = steps[top]
        if step.pos1:
            q.append(get_pos(step.pos1))
        if step.pos2:
            q.append(get_pos(step.pos2))

    return sorted(set(step_list))


# 归结推理 广度优先搜索
def resolve(clause_num: int, clause_set: [[Word]], steps: [Step]):
    end_flag: bool = False  # 是否到达 []
    end_step_idx: int = -1

    start_pos = 0  # 第二层遍历的起始位置
    end_pos = clause_num  # 第二层遍历的终止位置

    while not end_flag:  # 如果没有到达 []
        for i in range(end_pos):  # 第一层遍历：遍历所有的子句
            for j in range(start_pos, end_pos):  # 第二层遍历：遍历更新的子句
                if i < j:
                    end_flag, end_step_idx = unity(steps, clause_set, i, j)
                    if end_flag:
                        break
                    # 保证产生所有可能的归结情况
                    end_flag, end_step_idx = unity(steps, clause_set, j, i)
                    if end_flag:
                        break
            if end_flag:
                break
        # 更新第二层遍历的范围
        if len(clause_set) == end_pos:  # 如果子句没有增加，说明不能归结
            return []
        start_pos = end_pos
        end_pos = len(clause_set)
    # 回溯得到 归结推理的所有步骤
    return back_search(steps, end_step_idx)


def get_input(clause_num, clause_set):
    for i in range(clause_num):
        line = input().strip().replace(' ', '')
        if line.startswith('('):  # 子句
            pos = 1
            temp = ""
            while pos < len(line)-1:
                temp += line[pos]
                if line[pos] == ")":  # 说明是一个文字的结尾
                    clause_set[i].append(Word(temp))
                    pos += 1  # 跳过一个","号
                    temp = ""
                pos += 1
        else:  # 单个文字
            clause_set[i].append(Word(line))


def change_pos(pos: str, idx_map:{}):
    old_pos = get_pos(pos)
    new_pos = idx_map[old_pos]
    old_pos_str = str(old_pos)
    new_pos_str = str(new_pos)
    if len(old_pos_str) == len(pos):
        return new_pos_str
    return new_pos_str+pos[len(pos)-1]


def main():
    # 获取输入
    clause_num = int(input())
    clause_set = [[] for _ in range(clause_num)]
    get_input(clause_num, clause_set)
    # 初始化 steps 保证 更新得到 clause_set 的下标和对应步骤在 steps 中的下标一致
    steps = [Step("", "", [], [], []) for _ in range(clause_num)]
    # 归结推理
    resolve_steps = resolve(clause_num, clause_set, steps)
    if len(resolve_steps) == 0:
        print("无法归结")
        return

    idx_map = {}
    for i in range(clause_num):
        idx_map[i+1] = i+1

    cn = clause_num
    for i in resolve_steps:
        if i >= clause_num:
            cn += 1
            idx_map[i+1] = cn
            steps[i].pos1 = change_pos(steps[i].pos1, idx_map)
            steps[i].pos2 = change_pos(steps[i].pos2, idx_map)

    print("----------------- resolve_steps -----------------")
    # 输出归结推理的所有需要的步骤
    for i in resolve_steps:
        if i >= clause_num:
            print(str(steps[i]))


if __name__ == "__main__":
    main()
