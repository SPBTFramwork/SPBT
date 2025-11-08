from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang

def get_declare_info(node):
    # Returns all types of variable names in the node code block and the node dictionary
    type_ids_dict, type_dec_node = {}, {}
    declaration_map = {'c': 'declaration', 
                       'java': 'local_variable_declaration'}
    lang = get_lang()
    for child in node.children:
        if child.type == declaration_map[lang]:
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
    if node.type == 'identifier' and node.parent.type not in ['subscript_expression', 'call_expression']:   # a in a < 2
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)

def get_id_first_line(node):
    # Get the line number where all variables are first declared and used in the node code block
    first_declare, first_use = {}, {}
    declaration_map = {'c': 'declaration', 
                       'java': 'local_variable_declaration'}
    lang = get_lang()
    for child in node.children:
        if child.type == declaration_map[lang]:
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

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    try:
        while i >= 0 and code[i] != '\n':
            if code[i] == ' ':
                indent += 1
            elif code[i] == '\t':
                indent += 4
            i -= 1
    except:
        pass
    return indent

'''=========================match========================'''
def match_not_first(root):
    block_map = {'c': 'compound_statement',
                 'java': 'block'}
    declaration_map = {'c': 'declaration', 
                       'java': 'local_variable_declaration'}
    lang = get_lang()
    def check(node):
        # Defining variables is not concentrated in the first few lines
        is_first_dec = True
        if node.type == block_map[lang]:
            for child in node.children[1: -1]:
                if child.type != declaration_map[lang]:
                    if is_first_dec == True:
                        is_first_dec = False
                if child.type == declaration_map[lang] and is_first_dec == False:
                    return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_not_tmp(root):
    block_map = {'c': 'compound_statement',
                 'java': 'block'}
    lang = get_lang()
    def check(node):
        # The variable defined is not used in the previous line for the first time
        if node.type == block_map[lang]:
            first_declare, first_use = get_id_first_line(node)
            for id, dec in first_declare.items():
                if id in first_use and first_use[id] != dec + 1:
                    return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)
    return res

'''=========================replace========================'''          
def convert_first(node, code):
    # Put the variable name declaration at the front
    ret = []
    type_ids_dict, type_dec_node = get_declare_info(node)
    indent = get_indent(node.children[1].start_byte, code)
    start_byte = len(code)
    for type, node in type_dec_node.items():
        for each in node:
            ret.append((each.end_byte, each.start_byte))
            start_byte = min(start_byte, each.start_byte)
    declare_list = []
    for i, (type, ids) in enumerate(type_ids_dict.items()):
        if i == 0:
            declare_list.append(f"{type} {', '.join(ids)};")
        else:
            declare_list.append(f"{indent * ' '}{type} {', '.join(ids)};")
    ret.append((start_byte, '\n'.join(declare_list)))
    return ret

def count_first(root):
    nodes = match_not_tmp(root)
    return len(nodes)

def convert_temp(node, code):
    # Place the variable name declaration on the line before the first use of the variable name
    first_declare, first_use = get_id_first_line(node)
    declare_node, temp_id = [], []
    for id, dec in first_declare.items():
        if id in first_use and first_use[id] != dec + 1:
            temp_id.append(id)
    id_type_dict = {}
    declaration_map = {'c': 'declaration', 
                       'java': 'local_variable_declaration'}
    for child in node.children:
        if child.type == declaration_map[get_lang()]:
            type = text(child.children[0])
            for each in child.children[1: -1]:
                if each.type not in [',', ';']:
                    id_type_dict[text(each)] = type
                    if text(each) in temp_id and child not in declare_node:
                        declare_node.append(child)
    ret = []
    for each in declare_node:
        # First determine whether all the IDs in the node are in temp_id. If so, delete the entire row, otherwise only delete part of the ID.
        temp_id_node = []
        delete_all_line = True
        type = text(each.children[0])
        for ch in each.children[1: -1]:
            if ch.type not in [',', ';']:
                if text(ch) not in temp_id:
                    delete_all_line = False
                else:
                    temp_id_node.append(ch)
        # First delete the ids that are not declared in the line before the first use of the id.
        if delete_all_line == False:
            # Delete the id in the declare
            for id_node in temp_id_node:
                if id_node.next_sibling.next_sibling:  # If it is int a, b, c; a, b here are not the last elements
                    next_node = id_node.next_sibling.next_sibling
                    ret.append((next_node.start_byte, id_node.start_byte))
                elif id_node.next_sibling and id_node.next_sibling.type == ';':   # If it is the last element like c
                    prev_node = id_node.prev_sibling
                    ret.append((id_node.end_byte, prev_node.start_byte))
        else:   # Delete an entire line
            prev_node = each.prev_sibling
            ret.append((each.end_byte, prev_node.end_byte))
    # Then insert a row before all first uses of temp_id
    line_type_id_dict = {}  # Line number, type, variable name
    for id in temp_id:
        if id in id_type_dict.keys():
            type = id_type_dict[id]
            line = first_use[id]
            line_type_id_dict.setdefault(line, {})
            line_type_id_dict[line].setdefault(type, [])
            line_type_id_dict[line][type].append(id)
    for line in line_type_id_dict:
        for type in line_type_id_dict[line]:
            ids = line_type_id_dict[line][type]
            # Find the position of the corresponding line of line
            for child in node.children:
                if child.start_point[0] == line:
                    indent = get_indent(child.start_byte, code)
                    if len(ids) == 1:   # If you change the row and type of variable, only one is inserted.
                        dec_str = f"{type} {ids[0]};\n{indent * ' '}"
                    else:       # If there are multiple
                        dec_str = f"{type} {', '.join(ids)};\n{indent * ' '}"
                    ret.append((child.start_byte, dec_str))
    return ret

def count_temp(root):
    nodes = match_not_first(root)
    return len(nodes)