from ist_utils import replace_from_blob, traverse_rec_func, text, get_indent
from transform.lang import get_lang

def get_declare_info(node):
    # Returns all types of variable names in the node code block and the node dictionary
    lang = get_lang()
    variable_declaration_map = {'c': 'declaration', 
                                'java': 'local_variable_declaration'}
    type_ids_dict, type_dec_node = {}, {}
    for child in node.children:
        if child.type == variable_declaration_map[lang]:
            type = text(child.children[0])
            type_ids_dict.setdefault(type, [])
            type_dec_node.setdefault(type, [])
            type_dec_node[type].append(child)
            for each in child.children[1: -1]:
                if each.type == ',':
                    continue
                type_ids_dict[type].append(text(each))
    return type_ids_dict, type_dec_node

def contain_id(node, contain):
    # Returns all variable names in the subtree of node node
    if node.child_by_field_name('index'):   # index in a[i] < 2: i
        contain.add(text(node.child_by_field_name('index')))
    lang = get_lang()
    _map = {'c': ['init_declarator', 'declaration'], 
                 'java': ['variable_declarator']}
    # if node.type == 'identifier' and \
    #     node.parent.type in _map[lang] and \
    #     len(node.parent.children) >= 3 and \
    #     node.parent.children[0] == node:
    #     contain.add(text(node))
    if node.type == 'identifier' and \
        node.parent.type in _map[lang]:
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)

def get_id_first_line(node):
    # Get the line number of all variables first declared and used in the node code block
    first_declare, first_use = {}, {}
    for child in node.children:
        if child.type == 'declaration':
            dec_id = set()
            contain_id(child, dec_id)
            for each in dec_id:
                if each not in first_declare.keys():
                    first_declare[each] = child.start_point[0]
        # elif child.type not in ['if_statement', 'for_statement', 'else_clause', 'while_statement']: # Temporary variable names in compound statements are not considered
        else:
            use_id = set()
            contain_id(child, use_id)
            for each in use_id:
                if each not in first_use.keys():
                    first_use[each] = child.start_point[0]
    return first_declare, first_use

'''==========================match========================'''
def match_lines_merge(root):
    lang = get_lang()
    variable_declaration_map = {'c': 'declaration', 
                                'java': 'local_variable_declaration'}
    def check(node):
        # int a, b=0;
        if node.type == variable_declaration_map[lang]:
            ids = set()
            contain_id(node, ids)
            return len(ids) > 1
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)

    return res

def match_lines_split(root):
    lang = get_lang()
    variable_declaration_map = {'c': 'declaration', 
                                'java': 'local_variable_declaration'}
    def check(node):
        # int a; \n int b=0;
        type_list = set()
        for child in node.children:
            if child.type == variable_declaration_map[lang]:
                type = text(child.children[0])
                if type in type_list:
                    return True
                type_list.add(type)
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''=========================replace========================'''
def convert_lines_split(node, code):
    # int a, b; -> int a; int b;
    type = text(node.children[0])
    ret = [(node.end_byte, node.start_byte)]
    indent = get_indent(node.start_byte, code)
    for i, child in enumerate(node.children[1: -1]):
        if child.type == ',':
            continue
        ret.append((node.start_byte, f"\n{indent * ' '}{type} {text(child)};"))
    return ret

def count_lines_split(root):
    nodes = match_lines_split(root)
    return len(nodes)

def convert_lines_merge(node, code):
    # int a; int b; -> int a, b;
    ret = []
    indent = get_indent(node.children[1].start_byte, code)
    type_ids_dict, type_dec_node = get_declare_info(node)
    for type, ids in type_ids_dict.items():
        if len(ids) > 1:
            start_byte = type_dec_node[type][0].start_byte
            for node in type_dec_node[type]:
                ret.append((node.end_byte, node.start_byte))
            str = f"{type} {', '.join(type_ids_dict[type])};"
            ret.append((start_byte, str))
    return ret

def count_lines_merge(root):
    nodes = match_lines_merge(root)
    return len(nodes)