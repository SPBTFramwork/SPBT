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
def match_if_return(root):
    def find_return_node(u):
        if u.type == 'return_statement':
            global return_text
            return_text = text(u)
            return
        for v in u.children:
            find_return_node(v)
    global return_text
    return_text = None
    primitive_type = text(root.children[0].children[0])
    if primitive_type == 'void':
        return_text = 'return void();'
    else:
        find_return_node(root)
        if return_text is None:
            return_text = 'return void();'

    def check(node):
        if node.type != 'if_statement': return False
        if 'return' in text(node): return False
        else_node = None
        for u in node.children:
            if u.type == 'else_clause':
                else_node = u
        if else_node: return False
        return True
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''=========================replace========================'''
def cvt_return(node, code):
    opposite = {'==': '!=', '!=': '==', 
                '>': '<=', '<=': '>',
                '<': '>=', '>=': '<'}
    if_node = node.children[0]
    expr_node = node.children[1]
    expr_in_node = expr_node.children[1]

    while len(expr_in_node.children) >= 1 and text(expr_in_node.children[0]) == '(':
        expr_in_node = expr_in_node.children[1]

    if len(expr_in_node.children) == 2 and text(expr_in_node.children[0]) == '!':
        tmp_node = expr_in_node.children[1]
        if len(tmp_node.children) < 2: return
        while len(tmp_node.children) >= 1 and text(tmp_node.children[0]) == '(':
            tmp_node = tmp_node.children[1]
        expr = '(' + text(tmp_node) + ')'
    elif len(expr_in_node.children) == 3 and text(expr_in_node.children[1]) in list(opposite.keys()):
        expr = '(' + text(expr_in_node.children[0]) + \
                opposite[text(expr_in_node.children[1])] + \
                text(expr_in_node.children[2]) + ')'
    else:
        expr = '(!(' + text(expr_in_node) + '))'
        
    content_node = node.children[2]
    content = '\n'.join([text(content_node.children[u]) for u in range(1, len(content_node.children)-1)])
    global return_text
    indent = get_indent(if_node.start_byte, code)
    
    return [(content_node.end_byte, expr_node.start_byte),
            (expr_node.start_byte, expr + ' ' + return_text + '\n' + ' '*indent + content),]
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