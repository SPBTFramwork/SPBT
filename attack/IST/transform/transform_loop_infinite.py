from ist_utils import text, get_indent
from transform.lang import get_lang

declaration_map = {'c': 'declaration', 'java': 'local_variable_declaration'}
block_map = {'c': 'compound_statement', 'java': 'block'}

def get_for_info(node):
    # Extract the ABC information of the for loop, for(a; b; c) and the following statement
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [';', ')', declaration_map[get_lang()]]:
            if child.type == declaration_map[get_lang()]:
                abc[i] = child
            if child.prev_sibling and child.prev_sibling.type not in ['(', ';']:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ')' and i == 3:
            abc[3] = child
    return abc

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

'''=========================match========================'''

def match_for_while(root):
    def check(node):
        if node.type in ['for_statement', 'while_statement']:
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''=========================replace========================'''

def count_inf_while(root):
    nodes = match_for_while(root)
    res = 0
    for node in nodes:
        if node.type == 'for_statement':
            abc = get_for_info(node)
            if abc[1] and text(abc[1]) in ['1', 'true']:
                res += 1
        elif node.type == 'while_statement':
            conditional_node = node.children[1].children[1]
            if text(conditional_node) in ['1', 'true']:
                res += 1
    return res

def cvt_infinite_while(node, code):
    # a while(b) c
    if node.type == 'for_statement':
        res = []
        abc = get_for_info(node)
        child_index = 3 + (4 - abc.count(None)) - (abc[0] is not None and abc[0].type == declaration_map[get_lang()])
        res = [(node.children[child_index].end_byte, node.children[0].start_byte - node.children[child_index].end_byte)]    # Delete for(a; b; c)
        if abc[0] is not None:  # If there is a
            indent = get_indent(node.start_byte, code)
            if abc[0].type != declaration_map[get_lang()]:
                res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
            else:
                res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
        if abc[2] is not None and abc[3] is not None:  # If there is a c
            if abc[3].type == block_map[get_lang()]:     # Compound sentences are inserted in the first sentence of Yves Blake
                last_expression_node = abc[3].children[-2]
            else:       # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
                last_expression_node = abc[3]
            indent = get_indent(last_expression_node.start_byte, code)
            res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
        while_str = f"while(true)"
        res.append((node.children[0].start_byte, while_str))

        compound_node = node.children[-1]
        if len(compound_node.children) <= 1: return
        indent = get_indent(compound_node.children[1].start_byte, code)
        break_str = f"\n{' '*indent}if({text(abc[1]) if abc[1] else 'true'}) break;"
        res.append((compound_node.children[0].start_byte+1, break_str))

        return res
    elif node.type == 'while_statement':
        res = []
        compound_node = node.children[-1]
        if len(compound_node.children) <= 1: return
        indent = get_indent(compound_node.children[1].start_byte, code)
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
        conditional_node = node.children[1].children[1]
        opp_dfs(conditional_node)
        conditional_str = text(conditional_node)
        for key, val in mp.items():
            conditional_str = conditional_str.replace(key, val)
        
        break_str = f"\n{' '*indent}if({conditional_str}) break;"
        res.append((compound_node.children[0].start_byte+1, break_str))

        res.append((conditional_node.end_byte, conditional_node.start_byte))
        res.append((conditional_node.start_byte, 'true'))
        return res