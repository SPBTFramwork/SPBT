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
def match_if_add_else(root):
    def check(node):
        if node.type != 'if_statement': return False
        for v in node.children:
            if v.type == 'else_clause':
                return False
        return True
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res


'''=========================replace========================'''

def cvt_add_else(node, code):
    if_node = node.children[0]
    expr_node = node.children[1]
    compound_node = node.children[2]
    if len(compound_node.children) < 3:
        return
    expr_in_node = expr_node.children[1]
    new_str = ''
    opposite = {'==': '!=', '!=': '==', 
                '>': '<=',  '<=': '>',
                '<': '>=',  '>=': '<',
                '&&': '||', '||': '&&'}
    mp = {}
    def opp_dfs(u):
        while len(u.children) >= 2 and text(u.children[0]) == '(':
            u = u.children[1]
        if len(u.children) == 0:
            mp[text(u)] = '!' + text(u)
            return
        if '&&' not in text(u) and '||' not in text(u) and len(u.children) == 3:
            if text(u.children[1]) in opposite:
                mp[text(u)] = text(u.children[0]) + opposite[text(u.children[1])] + text(u.children[2])
            else:
                mp[text(u)] = '!' + text(u)
            return
        elif len(u.children) == 2 and text(u.children[0]) == '!':
            mp[text(u)] = text(u.children[1])
            return
        elif len(u.children) == 3:
            try:
                mp[text(u)] = text(u.children[0]) + opposite[text(u.children[1])] + text(u.children[2])
            except:
                mp[text(u)] = '!' + text(u)
        
        for v in u.children:
            opp_dfs(v)
    opp_dfs(expr_in_node)
    new_str = text(expr_in_node)
    for key, val in mp.items():
        new_str = new_str.replace(key, val)

    ok = 0
    next_node = None
    for v in node.parent.children:
        if ok and text(v) != '}':
            next_node = v
            break
        if v == node: ok = 1
    else_indent = get_indent(if_node.start_byte, code)
    compound_start_byte = compound_node.children[1].start_byte
    compound_end_byte = compound_node.children[len(compound_node.children) - 1].end_byte-1
    compound_str = ('\n' + ' ' * (else_indent+4)).join([text(compound_node.children[i]) for i in range(1, len(compound_node.children) - 1)])
    else_str = '\n%selse {\n%s%s\n%s}' % \
                (' '*else_indent, ' '*(else_indent+4), compound_str, ' '*else_indent)
    if next_node:
        return [(expr_in_node.end_byte, expr_in_node.start_byte),
                (expr_in_node.start_byte, new_str),
                (next_node.end_byte, next_node.start_byte),
                (compound_end_byte, compound_start_byte),
                (compound_start_byte, text(next_node) + '\n'),
                (node.end_byte, else_str)]
    else:
        return [(expr_in_node.end_byte, expr_in_node.start_byte),
                (expr_in_node.start_byte, new_str),
                (compound_end_byte, compound_start_byte),
                (node.end_byte, else_str)]

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