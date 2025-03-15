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

'''==========================match========================'''
def match_if_equivalent(root):
    def check(node):
        if node.type != 'if_statement': return False
        expr_node = node.children[1]
        expr_in_node = expr_node.children[1]
        while len(expr_in_node.children) >= 2 and text(expr_in_node.children[0]) == '(':
            expr_in_node = expr_in_node.children[1]
        if len(expr_in_node.children) < 3: return False
        return True
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''=========================replace========================'''
def cvt_equivalent(node, code):
    if_node = node.children[0]
    expr_node = node.children[1]
    expr_in_node = expr_node.children[1]
    new_str = ''
    opposite = {'==': '!=', '!=': '==', 
                '>': '<=', '<=': '>',
                '<': '>=', '>=': '<',
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
    new_str = '!(' + new_str + ')'
    return [(expr_in_node.end_byte, expr_in_node.start_byte),
            (expr_in_node.start_byte, new_str),]