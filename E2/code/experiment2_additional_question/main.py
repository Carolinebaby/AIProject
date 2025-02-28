import copy

pos_in_set = ['a', 'b', 'c', 'd', 'e', 'f']
variables = ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# ------------------------ 类 -------------------------
class Word:
    def __init__(self, word_str):
        self.arguments = []
        self.is_negative = False
        self.predicate = ""

        pos = 0
        if word_str[0] == '¬':
            self.is_negative = True
            pos = 1
        stack = []
        temp = ""

        while pos < len(word_str):
            if not len(stack):
                if word_str[pos] != "(":
                    temp += word_str[pos]
                else:  # 第一个"("，说明开始读入参数，前面的字符串为谓词
                    self.predicate = temp
                    temp = ""
                    stack.append("(")
            elif word_str[pos] == "(":  # 栈非空时的"("存入栈中，以区分函数
                stack.append("(")
                temp += word_str[pos]
            elif word_str[pos] == ")":
                stack.pop()
                if len(stack):  # 说明这是函数的 ")"
                    temp += word_str[pos]
                else:  # 说明到达了最后的 ")"
                    self.arguments.append(temp)
            elif word_str[pos] == ",":  # 说明前面的字符串为 argument
                self.arguments.append(temp)
                temp = ""
            else:
                temp += word_str[pos]
            pos += 1

    def __str__(self):
        arguments_str = ",".join(self.arguments)
        negative_str = "¬" if self.is_negative else ""
        return f"{negative_str}{self.predicate}({arguments_str})"

    def replace_var(self, old_name: [str], new_name: [str]):
        for i in range(len(old_name)):
            for j in range(len(self.arguments)):
                func_flag, var_pos = is_function(self.arguments[j])  # var_pos 返回的变量的地址
                if func_flag:  # 函数的情况
                    # 如果是变量，并且变量需要被替换
                    if is_var(self.arguments[j]) and self.arguments[j][var_pos] == old_name[i]:
                        # 替换变量
                        new_arg = ""
                        for k in range(0, var_pos):  # new_arg 先存入 原变量 前面的字符
                            new_arg += self.arguments[j][k]
                        new_arg += new_name[i]  # new_arg 加入 替换的新名字
                        for k in range(var_pos + 1, len(self.arguments[j])):  # 存入 原变量 后面的字符
                            new_arg += self.arguments[j][k]
                        self.arguments[j] = new_arg  # 改变 旧的参数
                elif self.arguments[j] == old_name[i]:  # 非函数情况，直接替换
                    self.arguments[j] = new_name[i]

    def __lt__(self, other):
        return self.predicate < other.predicate


class Step:
    def __init__(self, pos1, pos2, old_name: [], new_name: [], clause: [Word]):
        self.pos1 = pos1
        self.pos2 = pos2
        self.old_name = copy.deepcopy(old_name)
        self.new_name = copy.deepcopy(new_name)
        self.clause = clause

    def __str__(self):
        replace_str = ""
        if self.old_name:
            replace_str += "{" + ','.join(f"{old}={new}" for old, new in zip(self.old_name, self.new_name)) + "}"
        c_str = "[]" if not len(self.clause) else get_clause_str(self.clause)
        return f"R[{self.pos1},{self.pos2}]{replace_str} = {c_str}"


# ------------------------ 函数 -------------------------
def is_var(para: str):
    func_flag, var_pos = is_function(para)
    if func_flag:
        temp = ""
        while para[var_pos] != ")":
            temp += para[var_pos]
            var_pos += 1
        if temp in variables:
            return True, temp
    else:
        if para in variables:
            return True, para

    return False, ""


def is_function(arg):
    if len(arg) < 4:  # 字符串长度小于 4 直接排除
        return False, -1
    temp = ""
    pos = 0
    stack = []
    while pos < len(arg):
        if arg[pos] == "(":
            stack.append("(")
            temp = ""
        elif arg[pos] == ")":
            # 到达第一个 ),首先说明有函数，并且 ) 前面的字符串为变量或常量
            # temp 表示函数里面的 变量或常量
            # pos-len(temp) 表示 函数中的变量或常量所在的位置
            return True, pos-len(temp)
        else:
            temp += arg[pos]
        pos += 1
    return False, -1


def get_clause_str(clause: [Word]) -> str:
    clause_str = ""
    size = len(clause)
    for i in range(size):
        clause_str += str(clause[i])
        if i != size-1:
            clause_str += ","
    return clause_str


def is_same(w1: Word, w2: Word) -> bool and int:
    for i in range(len(w1.arguments)):
        if w1.arguments[i] != w2.arguments[i]:
            return False, i
    return True, 0


def change(pos: int, w1: Word, w2: Word):
    var_flag1, var_name1 = is_var(w1.arguments[pos])
    var_flag2, var_name2 = is_var(w2.arguments[pos])
    # 如果都不是变量，说明一定不能归结
    if not var_flag1 and not var_flag2:
        return False, "", ""

    func_flag1, var_pos1 = is_function(w1.arguments[pos])
    func_flag2, var_pos2 = is_function(w2.arguments[pos])

    if (not func_flag1 and not func_flag2 and var_flag1) or (not func_flag1 and func_flag2 and var_flag1):
        old_name = w1.arguments[pos]
        new_name = w2.arguments[pos]
        if func_flag2 and var_flag2 and (old_name == w2.arguments[pos][var_pos2]):  # 说明被替换的变量出现在了替换变量的项中
            return False, "", ""
        w1.replace_var([old_name], [new_name])
        return True, old_name, new_name
    elif (not func_flag1 and not func_flag2) or (func_flag1 and not func_flag2 and var_flag2):
        old_name = w2.arguments[pos]
        new_name = w1.arguments[pos]
        if func_flag1 and var_flag1 and (old_name == w1.arguments[pos][var_pos1]):
            return False, "", ""
        w2.replace_var([old_name], [new_name])
        return True, old_name, new_name
    elif (func_flag1 and not func_flag2) or (not func_flag1 and func_flag2):
        return False, "", ""
    else:  # 两个都是函数
        if len(w1.arguments[pos]) > len(w2.arguments[pos]):
            # 第一个参数字符串更长
            if (var_flag1 and not var_flag2) or (w1.arguments[pos][var_pos1] == w2.arguments[pos][var_pos2]):
                # 如果第一个函数是变量，第二个函数不是变量
                # 说明再怎么改变都无法使第一个参数变成第二个参数
                # 如果第一个函数的变量和第二个函数的变量相同，则无法归结
                return False, "", ""

            # 说明 w2.arg[pos]是变量，可以改变 w2.arg[pos] 使参数一致
            i = 0
            while i < min(len(w1.arguments[pos]), len(w2.arguments[pos])):
                if w1.arguments[pos][i] != w2.arguments[pos][i]:
                    break
                i += 1
            # i 到达第一个不同的地方,如果w2.arg[pos]这个地方是变量
            # 说明，w2的变量可替换，否则说明w2变量前面就出现了不同
            # 则不可替换
            if i != var_pos2:
                return False, "", ""

            if w2.arguments[pos][i] in variables:
                old_name = w2.arguments[pos][i]
                new_name = ""
                stack = []
                # 利用栈类型，通过匹配括号，得到 w2.arg[pos]的变量可以替换的
                # w1.arg[pos] 中的对应字符串
                while i < len(w1.arguments[pos]):
                    new_name += w1.arguments[pos][i]
                    if w1.arguments[pos][i] == "(":
                        stack.append("(")
                    elif w1.arguments[pos][i] == ")":
                        stack.pop()
                        if not len(stack):
                            break
                    i += 1
                w2.replace_var([old_name], [new_name])
                return True, old_name, new_name

        elif len(w2.arguments[pos]) > len(w1.arguments[pos]):
            if (var_flag2 and not var_flag1) or (w1.arguments[pos][var_pos1] == w2.arguments[pos][var_pos2]):
                return False, "", ""
            i = 0
            while i < min(len(w2.arguments[pos]), len(w1.arguments[pos])):
                if w2.arguments[pos][i] != w1.arguments[pos][i]:
                    break
                i += 1
            if i != var_pos1:
                return False, "", ""

            if w1.arguments[pos][i] in variables:
                old_name = w1.arguments[pos][i]
                new_name = ""
                stack = []
                while i < len(w2.arguments[pos]):
                    new_name += w2.arguments[pos][i]
                    if w2.arguments[pos][i] == "(":
                        stack.append("(")
                    elif w2.arguments[pos][i] == ")":
                        stack.pop()
                        if not len(stack):
                            break
                    i += 1
                if old_name in new_name:
                    return False, "", ""
                w1.replace_var([old_name], [new_name])
                return True, old_name, new_name

        else:  # 两个参数的长度相等，说明函数的参数都是一个字符长度
            i = 0
            while i < len(w1.arguments[pos]):
                if w1.arguments[pos][i] != w2.arguments[pos][i]:
                    break
                i += 1
            if i != var_pos1:  # 如果不同的地方不是变量所在的地方，说明前面的函数就出现了不同
                return False, "", ""
            if var_flag1:
                old_name = w1.arguments[pos][var_pos1]
                new_name = w2.arguments[pos][var_pos2]
                w1.replace_var([old_name], [new_name])
                return True, old_name, new_name
            else:
                old_name = w2.arguments[pos][var_pos2]
                new_name = w1.arguments[pos][var_pos1]
                w2.replace_var([old_name], [new_name])
                return True, old_name, new_name

    return False, "", ""


def judge(w1: Word, w2: Word):
    word1 = copy.deepcopy(w1)
    word2 = copy.deepcopy(w2)
    old_name = []
    new_name = []
    if word1.predicate != word2.predicate or len(word1.arguments) != len(word2.arguments) or (word1.is_negative and word2.is_negative) or (not word1.is_negative and not word2.is_negative):
        return False, old_name, new_name

    flag, dif_pos = is_same(word1, word2)
    while not flag:
        is_changed, old, new = change(dif_pos, word1, word2)
        if not is_changed:
            return False, old_name, new_name
        old_name.append(old)
        new_name.append(new)
        flag, dif_pos = is_same(word1, word2)

    return True, old_name, new_name


def union(clause1: [Word], clause2: [Word], pos1, pos2, old_name: [str], new_name: [str]) -> [Word]:
    c1 = copy.deepcopy(clause1)
    c2 = copy.deepcopy(clause2)
    for clause in c1:
        clause.replace_var(old_name, new_name)

    for clause in c2:
        clause.replace_var(old_name, new_name)

    uc = []
    for i in range(len(c1)):
        if i != pos1:
            new_flag = True
            for word in uc:
                if str(c1[i]) == str(word):
                    new_flag = False
            if new_flag:
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
    for c in clause_set:
        if get_clause_str(clause) == get_clause_str(c):
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

                if not uc:
                    end_flag = True
                    end_step_idx = len(steps) + 1

                if is_exist(uc, clause_set):
                    continue
                clause_set.append(uc)
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

    start_pos = 0
    end_pos = clause_num

    while not end_flag:
        for i in range(end_pos):
            for j in range(start_pos, end_pos):
                if i < j:
                    end_flag, end_step_idx = unity(steps, clause_set, i, j)
                    if end_flag:
                        break
                    end_flag, end_step_idx = unity(steps, clause_set, j, i)
                    if end_flag:
                        break
            if end_flag:
                break
        if len(clause_set) == end_pos:
            return []
        start_pos = end_pos
        end_pos = len(clause_set)
    return back_search(steps, end_step_idx)


def get_input(clause_num, clause_set):
    for i in range(clause_num):
        line = input().strip().replace(' ', '')
        if line.startswith('('):   # 子句
            pos = 1  # pos : 1~len(line)-2, 排除了子句前后的括号
            temp = ""
            stack = []  # 利用栈区分 参数内部函数 和 谓词后面的括号
            clause = [] # 子句
            while pos < len(line)-1:
                if not len(stack):
                    if line[pos] == '(':
                        stack.append("(")
                    temp += line[pos]
                elif line[pos] == '(':
                    stack.append(")")
                    temp += line[pos]
                elif line[pos] == ')':
                    stack.pop()
                    temp += line[pos]
                    if not len(stack):
                        if temp[0] == ',':
                            temp = temp[1:]
                        clause.append(Word(temp))
                        temp = ""
                elif line[pos] == ',' and not len(stack):
                    continue
                else:
                    temp += line[pos]
                pos += 1
            clause_set[i] = clause
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
    clause_num = int(input())
    clause_set = [[] for _ in range(clause_num)]
    get_input(clause_num, clause_set)
    steps = [Step("", "", [], [], []) for _ in range(clause_num)]

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
    for i in resolve_steps:
        if i >= clause_num:
            print(str(steps[i]))


if __name__ == "__main__":
    main()
