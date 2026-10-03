# ai.py
import ast
import numpy as np


def analyze_var_names(code):
    tree = ast.parse(code)
    variables = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    name_len = [len(name) for name in variables]
    avg_name_len = np.mean(name_len) if name_len else 0

    return avg_name_len > 5


c = """
#n, k = map(int, input())
#a = list(map(int, input().split()))

funcs = {}

def run_code(code):
    if code != False:
        #print('> called func')
        code.append('endprg')
    runpointer = 0
    infunc = 0
    currentwritingfunc = ""
    while True:
        curr_op = ""
        if code == False:
            curr_op = input()
            #print("> running at input")
        else:
            curr_op = code[runpointer]
            runpointer += 1

        if curr_op == "endprg": break

        if curr_op[0:8] == "function":
            infunc += 1
            currentwritingfunc = curr_op[9:len(curr_op)]
            funcs[currentwritingfunc] = [1, []]
            #print("> doing func", currentwritingfunc)
            continue

        if curr_op[0:3] == "end":
            infunc -= 1
            continue

        if infunc > 0:
            funcs[currentwritingfunc][1].append(curr_op)
            #print('> wrote \"'+curr_op+'\" to '+currentwritingfunc)
        else:
            if curr_op[0:4] == "call":
                #print(funcs)
                run_code(funcs[curr_op[5:len(curr_op)]][1])
                continue
            if curr_op[0:5] == "print":
                print(curr_op[6:len(curr_op)])
                continue

run_code(False)
"""

print(analyze_var_names(c))