from ist_utils import replace_from_blob, traverse_rec_func, text, print_children, get_indent
from transform.lang import get_lang

def get_declare_info(node):
    # Returns all types of variable names in the node code block and the node dictionary
    type_ids_dict, type_dec_node = {}, {}
    for child in node.children:
        if child.type == 'declaration':
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

'''=========================match========================'''

def match_assign_merge(root):
    if get_lang() == 'java':
        def check(node):
            if node.type == 'local_variable_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator' and len(child.children) == 3:
                        return True
            return False
    elif get_lang() == 'c_sharp':
        def check(node):
            if node.type == 'local_declaration_statement':
                for child in node.children:
                    if child.type == 'variable_declaration':
                        return True
            return False
    elif get_lang() == 'c':
        def check(node):
            if node.type == 'declaration':
                for child in node.children:
                    if child.type == 'init_declarator':
                        return True
            return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_assign_split(root):
    if get_lang() == 'c':
        def check(node):
            if node.type == 'declaration':
                for child in node.children:
                    if child.type == 'init_declarator':
                        return False
                return True
            return False
    elif get_lang() == 'c_sharp':
        def check(node):
            if node.type == 'local_declaration_statement':
                for child in node.children:
                    if child.type == 'variable_declaration' and len(child.children) == 3:
                        return False
                return True
            return False
    elif get_lang() == 'java':
        def check(node):
            if node.type == 'local_variable_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator' and len(child.children) == 3:
                        return False
                return True
            return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def convert_assign_split(node, code):
    # int i = 0; -> int i; \n i = 0;
    if node.parent.type == 'for_statement': return
    ret = []
    if get_lang() == 'c':
        for child in node.children:
            if child.type == 'init_declarator':
                declarator = child.child_by_field_name('declarator')
                value = child.child_by_field_name('value')
                indent = get_indent(node.start_byte, code)
                ret.append((value.end_byte, declarator.end_byte))
                ret.append((node.end_byte, f"\n{indent*' '}{text(declarator)} = {text(value)};"))
    elif get_lang() == 'c_sharp':
        for child in node.children:
            if child.type == 'variable_declaration':
                for u in child.children:
                    if u.type == 'variable_declarator':
                        declarator = u.children[0]
                        if len(u.children) < 2: return
                        if len(u.children[1].children) < 2: return
                        value = u.children[1].children[1]
                        indent = get_indent(node.start_byte, code)
                        ret.append((value.end_byte, declarator.end_byte))
                        ret.append((node.end_byte, f"\n{indent*' '}{text(declarator)} = {text(value)};"))
    elif get_lang() == 'java':
        for child in node.children:
            if child.type == 'variable_declarator':
                if len(child.children) <= 2: return
                declarator = child.children[0]
                value = child.children[2]
                indent = get_indent(node.start_byte, code)
                ret.append((value.end_byte, declarator.end_byte))
                ret.append((node.end_byte, f"\n{indent*' '}{text(declarator)} = {text(value)};"))
    return ret

def count_assign_split(root):
    nodes = match_assign_split(root)
    return len(nodes)

def convert_assign_merge(node, code):
    # int i; i = 0; -> int i = 0;
    type_node = node.children[0]
    var_node = node.children[1]
    assign_nodes = []

    def find_val_node(u):
        if len(assign_nodes) >= 1:
            return
        if u.type == 'assignment_expression' and text(u.children[0]) == text(var_node):
            assign_nodes.append(u)
            return
        for v in u.children:
            find_val_node(v)
    find_val_node(node.parent)
    
    if len(assign_nodes) == 0:
        return
    assign_node = assign_nodes[0]
    val_node = assign_node.children[2]

    return [(node.end_byte, node.start_byte),
            (assign_node.end_byte+1, assign_node.start_byte),
            (node.start_byte, f"{text(type_node)} {text(var_node)} = {text(val_node)};")]

def count_assign_merge(root):
    nodes = match_assign_merge(root)
    return len(nodes)