from ist_utils import text, print_children
from transform.lang import get_lang

def match_cmp(root):
    # a ? b
    res = []
    def check(u):
        if u.type == 'binary_expression' and \
            text(u.children[1]) in ['>', '>=', '<', '<=', '==', '!='] and \
            len(u.children) == 3:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def match_bigger(root):
    # a >/>= b
    res = []
    def check(u):
        if u.type == 'binary_expression' \
                and text(u.children[1]) in ['>', '>=']:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res
    
def match_smaller(root):
    # a </<= b
    res = []
    def check(u):
        if u.type == 'binary_expression' \
                and text(u.children[1]) in ['<', '<=']:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def match_equal(root):
    # a </<= b
    res = []
    def check(u):
        if u.type == 'binary_expression' \
                and text(u.children[1]) in ['==']:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def match_not_equal(root):
    # a </<= b
    res = []
    def check(u):
        if u.type == 'binary_expression' \
                and text(u.children[1]) in ['!=']:
            return True
        return False

    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def convert_smaller(node):
    [a, op, b] = [text(x) for x in node.children]
    if op in ['<=', '<']: return
    if op in ['>=', '>']:
        reverse_op_dict = {'>':'<', '>=':'<='}
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{b} {reverse_op_dict[op]} {a}')]
    if op in ['==']:
        # b <= a && a <= b
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} <= {b} && {b} <= {a}')]
    if op in ['!=']:
        # a < b || b < a
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} < {b} || {b} < {a}')]

def convert_bigger(node):
    [a, op, b] = [text(x) for x in node.children]
    if op in ['>=', '>']: return
    if op in ['<=', '<']:
        reverse_op_dict = {'<':'>', '<=':'>='}
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{b} {reverse_op_dict[op]} {a}')]
    if op in ['==']:
        # a >= b && b >= a
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} >= {b} && {b} >= {a}')]
    if op in ['!=']:
        # a > b || b > a
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} > {b} || {b} > {a}')]

def convert_equal(node):
    [a, op, b] = [text(x) for x in node.children]
    if op in ['==']: return
    if op in ['<=']:
        # a < b || a == b
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} < {b} || {a} == {b}')]
    if op in ['<']:
        # !(b < a || a == b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({b} < {a} || {a} == {b})')]
    if op in ['>=']:
        # a > b || a == b
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} > {b} || {a} == {b}')]
    if op in ['>']:
        # !(b > a || a == b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({b} > {a} || {a} == {b})')]
    if op in ['!=']:
        # !(a == b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({a} == {b})')]

def convert_not_equal(node):
    [a, op, b] = [text(x) for x in node.children]
    if op in ['!=']: return
    if op in ['<=']:
        # !(b < a && a != b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({b} < {a} && {a} != {b})')]
    if op in ['<']:
        # (a < b && a != b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} < {b} && {a} != {b}')]
    if op in ['>=']:
        # !(b > a && a != b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({b} > {a} && {a} != {b})')]
    if op in ['>']:
        # a < b && a != b
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'{a} < {b} && {a} != {b}')]
    if op in ['==']:
        # !(a != b)
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f'!({a} != {b})')]

def count_bigger(root):
    nodes = match_bigger(root)
    return len(nodes)

def count_smaller(root):
    nodes = match_smaller(root)
    return len(nodes)

def count_equal(root):
    nodes = match_equal(root)
    return len(nodes)

def count_not_equal(root):
    nodes = match_not_equal(root)
    return len(nodes)