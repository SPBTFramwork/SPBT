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

def match_if_split(root):
    def check(node):
        if node.type != 'if_statement': return False
        if get_lang() == 'c_sharp':
            expr_in_node = node.children[2]
        else:
            expr_node = node.children[1]
            if len(expr_node.children) <= 1: return
            expr_in_node = expr_node.children[1]
        
        while len(expr_in_node.children) >= 2 and text(expr_in_node.children[0]) == '(':
            expr_in_node = expr_in_node.children[1]
        if len(expr_in_node.children) < 3: return False
        # if text(expr_in_node.children[1]) != '&&': return False
        return True
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''==========================replace========================'''
def cvt_split(node):
    if_node = node.children[0]
    if get_lang() == 'c_sharp':
        expr_in_node = node.children[2]
    else:
        expr_node = node.children[1]
        expr_in_node = expr_node.children[1]
    while len(expr_in_node.children) >= 2 and text(expr_in_node.children[0]) == '(':
        expr_in_node = expr_in_node.children[1]

    def dfs(u):
        if len(u.children) != 3 or text(u.children[1]) != '&&':
            while len(u.children) >= 2 and text(u.children[0]) == '(':
                u = u.children[1]
            return text(u)
        l = [dfs(u.children[0]), dfs(u.children[2])]
        return ';'.join(l)
    expr_list = dfs(expr_in_node).split(';')
    if_expr_list = []
    for expr in expr_list:
        if_expr_list.append(f'if({expr})')
    if len(if_expr_list) < 2:
        if_expr_list.append("if(1)")
        return
    new_str = ' '.join(if_expr_list)
    
    if get_lang() == 'c_sharp':
        return [(node.children[3].end_byte, if_node.start_byte),
                (if_node.start_byte, new_str),]
    else:
        return [(expr_node.end_byte, if_node.start_byte),
                (if_node.start_byte, new_str),]

def count_if_split(root):
    return 0