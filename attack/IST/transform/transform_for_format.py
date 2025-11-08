from ist_utils import text
from transform.lang import get_lang

declaration_map = {'c': 'declaration', 'java': 'local_variable_declaration', 'c_sharp': 'local_variable_declaration'}
block_map = {'c': 'compound_statement', 'java': 'block', 'c_sharp': 'block'}

def get_for_info(node):
    # Extract the ABC information of the for loop, for(a; b; c) and the following statement
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [';', ')', declaration_map[get_lang()]]:
            if child.type == declaration_map[get_lang()]:
                abc[i] = child
            if child.prev_sibling is None: return
            if child.prev_sibling.type not in ['(', ';']:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ')' and i == 3:
            abc[3] = child
    return abc

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    if len(code) <= i: return indent
    while i >= 0 and code[i] != '\n':
        if code[i] == ' ':
            indent += 1
        elif code[i] == '\t':
            indent += 4
        i -= 1
    return indent

def contain_id(node, contain):
    # Returns the names of all variables in the node subtree
    if node.child_by_field_name('index'):   # a[i] index in < 2: i
        contain.add(text(node.child_by_field_name('index')))
    if node.type == 'identifier' and node.parent.type not in ['subscript_expression', 'call_expression']:   # A in a < 2
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)

'''==========================matching========================'''
def rec_For(node):
    if node.type == 'for_statement':
        return True

def match_for(root):
    def check(node):
        if node.type == 'for_statement':
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''==========================replacement========================'''
def convert_abc(node, code):
    return

def count_abc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is not None and abc[1] is not None and abc[2] is not None
    return res

def convert_obc(node, code):
    # a for(;b;c)
    abc = get_for_info(node)
    indent = get_indent(node.start_byte, code)
    if abc[0] is not None:  # If there is a
        if abc[0].type != declaration_map[get_lang()]:    
            return [(abc[0].end_byte, abc[0].start_byte),
                    (node.start_byte, text(abc[0]) + f';\n{indent * " "}')]
        else:   # If int a, b is inside the for loop
            return [(abc[0].end_byte - 1, abc[0].start_byte),
                    (node.start_byte, text(abc[0]) + f'\n{indent * " "}')]

def count_obc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is None and abc[1] is not None and abc[2] is not None
    return res

def convert_aoc(node, code):
    # for(a;;c) if b break
    res, add_bracket = [], None
    abc = get_for_info(node)
    if abc[1] is not None:  # If there is a b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # Compound statements are inserted in the first sentence
            first_expression_node = abc[3].children[1]
        else:       # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append((first_expression_node.start_byte, f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}"))
        if add_bracket:
            res.extend(add_bracket)
        return res

def count_aoc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is not None and abc[1] is None and abc[2] is not None
    return res
             
def convert_abo(node, code):
    # for(a;b;) c
    res, add_bracket = [], None
    abc = get_for_info(node)   
    if abc[2] is not None:  # If there is a c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # Compound statements are inserted in the first sentence
            last_expression_node = abc[3].children[-2]
        else:       #If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
        if add_bracket:
            res.extend(add_bracket)
        return res

def count_abo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is not None and abc[1] is not None and abc[2] is None
    return res

def convert_aoo(node, code):
    # for(a;;) if b break c
    res, add_bracket = [], None
    abc = get_for_info(node)
    if abc[1] is not None:  # If there is a b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # Compound statements are inserted in the first sentence
            first_expression_node = abc[3].children[1]
        else:       # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append((first_expression_node.start_byte, f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}"))
    if abc[2] is not None:  # If there is a c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3].type == block_map[get_lang()]:     # Compound statements are inserted in the first sentence
            last_expression_node = abc[3].children[-2]
        else:       # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res

def count_aoo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is not None and abc[1] is None and abc[2] is None
    return res

def convert_obo(node, code):
    # a for(;b;) c
    res, add_bracket = [], None
    abc = get_for_info(node)
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != declaration_map[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))  
    if abc[2] is not None:  # If there is c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # The compound statement inserts if b break in the first sentence
            last_expression_node = abc[3].children[-2]
        else:       # If it is a single line, followed by curly braces, insert it at the beginning of expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res

def count_obo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is None and abc[1] is not None and abc[2] is None
    return res

def convert_ooc(node, code):
    # a for(;;c) if b break
    res, add_bracket = [], None
    abc = get_for_info(node)
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != declaration_map[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
    if abc[1] is not None:  #If there is b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # The compound statement inserts if b break in the first sentence
            first_expression_node = abc[3].children[1]
        else:       # If it is a single line, followed by curly braces, insert it at the beginning of expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append((first_expression_node.start_byte, f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}"))
    if add_bracket:
        res.extend(add_bracket)
    return res

def count_ooc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is None and abc[1] is None and abc[2] is not None
    return res

def convert_ooo(node, code):
    # a for(;;;) if break b c
    res, add_bracket = [], None
    abc = get_for_info(node)
    if abc is None: return
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != declaration_map[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
    if abc[1] is not None:  # If there is b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if abc[3].type == block_map[get_lang()]:     # The compound statement inserts if b break in the first sentence
            first_expression_node = abc[3].children[1]
        else:       # If it is a single line, followed by curly braces, insert it at the beginning of expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append((first_expression_node.start_byte, f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}"))
    if abc[2] is not None:  # If there is c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3].type == block_map[get_lang()]:     # The compound statement inserts if b break in the first sentence
            last_expression_node = abc[3].children[-2]
        else:       # If it is a single line, followed by curly braces, insert it at the beginning of expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res

def count_ooo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info(_node)
        res += abc[0] is None and abc[1] is None and abc[2] is None
    return res