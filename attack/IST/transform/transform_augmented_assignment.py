from ist_utils import text
from transform.lang import get_lang

def match_augmented_assignment(root):
    # a ?= b
    res = []
    augmented_assignments = ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=', '~=']
    def check(u):
        if u.type == 'assignment_expression' and u.child_count == 3 and \
            text(u.children[1]) in augmented_assignments:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)
    return res

def match_non_augmented_assignment(root):
    # a = b ? c ? a ? d ? e
    res = []
    ops = ['+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^', '~']
    def check(u):
        if u.type != 'assignment_expression': return False
        main_var = u.children[0]
        calc_expr = u.children[2]
        if len(calc_expr.children) < 3: return False
        if text(calc_expr.children[1]) not in ops: return False
        if text(main_var) == text(calc_expr.children[0]) or \
            text(main_var) == text(calc_expr.children[2]): return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def convert_non_augmented_assignment(node):
    # a ?= b -> a = a ? b
    [a, op, b] = [text(x) for x in node.children]
    new_str = f'{a} = {a} {op[:-1]} {b}'
    return [(node.end_byte, node.start_byte),
            (node.start_byte, new_str)] 

def count_non_augmented_assignment(root):
    nodes = match_non_augmented_assignment(root)
    return len(nodes)

def convert_augmented_assignment(node):
    # a = a ? b -> a ?= b
    main_var  = node.children[0]
    calc_expr = node.children[2]
    op = calc_expr.children[1]
    if text(calc_expr.children[0]) == text(main_var):
        new_str = f'{text(main_var)} {text(op)}= {text(calc_expr.children[2])}'
    else:
        new_str = f'{text(main_var)} {text(op)}= {text(calc_expr.children[0])}'
    return [(node.end_byte, node.start_byte),
            (node.start_byte, new_str)]

def count_augmented_assignment(root):
    nodes = match_augmented_assignment(root)
    return len(nodes)