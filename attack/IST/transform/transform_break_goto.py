from ist_utils import text
from transform.lang import get_lang

def get_for_info(node):
    # Extract the abc information of the for loop, for(a;b;c) and the following statements
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [';', ')', 'declaration']:
            if child.type == 'declaration':
                abc[i] = child
            if child.prev_sibling.type not in ['(', ';']:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ')' and i == 3:
            abc[3] = child
    return abc

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    while i >= 0 and code[i] != '\n':
        if code[i] == ' ':
            indent += 1
        elif code[i] == '\t':
            indent += 4
        i -= 1
    return indent

def contain_id(node, contain):
    # Returns all variable names in the subtree of node node
    if node.child_by_field_name('index'):   # index in a[i] < 2: i
        contain.add(text(node.child_by_field_name('index')))
    if node.type == 'identifier' and node.parent.type not in ['subscript_expression', 'call_expression']:   # a in a < 2
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)

'''=========================match========================'''

def match_break(root):
    def check(node):
        if node.type == 'break_statement' and \
            node.parent and node.parent.parent and \
            node.parent.parent.parent:
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''==========================replace========================'''
def cvt_break2goto(node, code):
    indent = get_indent(node.parent.parent.parent.start_byte, code)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, "goto endLoop;"),
            (node.parent.parent.parent.end_byte, f"\n{' '*(indent-1)}endLoop:\n")]