from ist_utils import text, print_children
from transform.lang import get_lang

return_text = None

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    while i >= 0 and code[i] != '\n':
        if code[i] == ' ':
            indent += 1
        elif code[i] == '\t':
            indent += 8
        i -= 1
    return indent

'''=========================match========================'''
def match_switch(root):
    switch_mapping = {'c': 'switch_statement', 'java': 'switch_expression'}
    def check(node):
        if node.type != switch_mapping[get_lang()]: return False
        return True
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_if(root):
    def check(node):
        if node.type != 'if_statement': return False
        if len(node.children) < 2: return False
        if len(node.children[1].children) < 2: return False
        if len(node.children[1].children[1].children) < 1: return False
        var_node = node.children[1].children[1].children[0]
        ok = [True]
        def check_eq(u):
            if not ok[0]: return
            if u.type == 'if_statement':
                v = u.children[1].children[1]
                if len(v.children) < 2: return False
                if text(v.children[1]) != '==' or \
                    text(var_node) not in [text(v.children[0]), text(v.children[2])]:
                    ok[0] = False
                    return
            for v in u.children:
                check_eq(v)
        check_eq(node)
        return ok[0]
    
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    vis = [0 for _ in range(len(res))]
    for i in range(len(res)):
        for j in range(len(res)):
            if i != j and text(res[i]) in text(res[j]):
                vis[i] = 1
    nres = []
    for i in range(len(res)):
        if not vis[i]:
            nres.append(res[i])
    return nres

'''==========================replace========================'''
def cvt_switch2if(node, code):
    var_str = text(node.children[1].children[1])
    compound_node = node.children[2]
    case_nodes = []
    for v in compound_node.children:
        if v.type == 'case_statement':
            case_nodes.append(v)
    if_strs = []
    indent = get_indent(node.start_byte, code)
    for i, case_node in enumerate(case_nodes):
        condition_str = text(case_node.children[1])
        if len(case_node.children) >= 4 and text(case_node.children[0]) == 'case':
            main_str = text(case_node.children[3])
        elif len(case_node.children) >= 3 and text(case_node.children[0]) == 'default':
            main_str = text(case_node.children[2])
        else:
            main_str = ''
        
        if i == 0:
            if_strs.append(f"{' '*indent}if({var_str} == {condition_str}){{\n{' '*indent*2}{main_str}\n{' '*indent}}}")
        elif i == len(case_nodes) - 1:
            if_strs.append(f"{' '*indent}else{{\n{' '*indent*2}{main_str}\n{' '*indent}}}")
        else:
            if_strs.append(f"{' '*indent}else if({var_str} == {condition_str}){{\n{' '*indent*2}{main_str}\n{' '*indent}}}")
    
    return [(node.end_byte, node.start_byte),
            (node.start_byte, '\n'.join(if_strs))]
    
def cvt_if2switch(node, code):
    # print_children(node)
    var_node = node.children[1].children[1].children[0]
    val_nodes = []
    block_nodes = []
    default_node = []
    def find_nodes(u):
        if u.type == 'if_statement':
            cond_node = u.children[1].children[1]
            if len(cond_node.children) <= 2: return
            if text(var_node) == text(cond_node.children[0]):
                val_nodes.append(cond_node.children[2])
            else:
                val_nodes.append(cond_node.children[0])
            block_nodes.append(u.children[2])
        if u.type == 'else_clause' and u.children[1].type != 'if_statement':
            default_node.append(u.children[1])
        for v in u.children:
            find_nodes(v)
    find_nodes(node)
    if len(default_node) == 0: return
    default_node = default_node[0]
    case_str = ""
    if_indent = get_indent(node.start_byte, code)
    for i in range(len(val_nodes)):
        tmp_str = ''
        for j, v in enumerate(block_nodes[i].children):
            if j == 0:
                tmp_str += f"{text(v)}\n"
            elif j == len(block_nodes[i].children) - 1:
                tmp_str += f"{' '*if_indent*3}break;\n{' '*if_indent*2}{text(v)}\n"
            else:
                tmp_str += f"{' '*if_indent*3}{text(v)}\n"
        case_str += f"{' '*if_indent*2}case {text(val_nodes[i])}:{tmp_str}\n"
    tmp_str = ''
    for j, v in enumerate(default_node.children):
        if j == 0:
            tmp_str += f"{text(v)}\n"
        elif j == len(default_node.children) - 1:
            tmp_str += f"{' '*if_indent*3}\n{' '*if_indent*2}{text(v)}\n"
        else:
            tmp_str += f"{' '*if_indent*3}{text(v)}\n"
    case_str += f"{' '*if_indent*2}default:{tmp_str}"
    switch_str = f"{' '*if_indent}switch({text(var_node)}){{\n{case_str}{' '*if_indent}}}"
    return [(node.end_byte, node.start_byte),
            (node.start_byte, switch_str)]


def count_switch(root):
    nodes = match_switch(root)
    return len(nodes)

def count_if(root):
    nodes = match_if(root)
    return len(nodes)