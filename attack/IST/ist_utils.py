import re
from tree_sitter import Parser, Language
import inspect

cpp_keywords = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm',
    'auto', 'bitand', 'bitor', 'bool', 'break',
    'case', 'catch', 'char', 'char16_t', 'char32_t',
    'class', 'compl', 'const', 'constexpr', 'const_cast',
    'continue', 'decltype', 'default', 'delete', 'do',
    'double', 'dynamic_cast', 'else', 'enum', 'explicit',
    'export', 'extern', 'false', 'float', 'for',
    'friend', 'goto', 'if', 'inline', 'int',
    'long', 'mutable', 'namespace', 'new', 'noexcept',
    'not', 'not_eq', 'nullptr', 'operator', 'or',
    'or_eq', 'private', 'protected', 'public', 'register',
    'reinterpret_cast', 'return', 'short', 'signed', 'sizeof',
    'static', 'static_assert', 'static_cast', 'struct', 'switch',
    'template', 'this', 'thread_local', 'throw', 'true',
    'try', 'typedef', 'typeid', 'typename', 'union',
    'unsigned', 'using', 'virtual', 'void', 'volatile',
    'wchar_t', 'while', 'xor', 'xor_eq'
]

text = lambda x: x.text.decode('utf-8')

def parent(node, dep):
    for _ in range(dep):
        node = node.parent
    return node

def get_parameter_count(func):
    signature = inspect.signature(func)
    params = signature.parameters
    return len(params)

def replace_from_blob(operation, blob):
    diff = 0        # The length difference caused by the insertion deletion
    # Sort the first element and the modified position from smallest to largest, if there is the same location to delete and insert, delete first and then insert
    operation = sorted(operation, key=lambda x: (x[0], 1 if type(x[1]) is int else 0, -len(x[1]) if type(x[1]) is not int else 0)) # The first key is index, the second key is deleted first, and the third key is inserted first with fewer strings
    for op in operation:
        if type(op[1]) is int:  # If the second element is a number
            if op[1] < 0:      # If it is less than 0, the elements are removed from op[0] to the left to op[1].
                del_num = op[1]
            else:       # then remove the elements from op[0] to the left to op[1].
                del_num = op[1] - op[0]
            blob = blob[:op[0] + diff + del_num] + blob[op[0] + diff:]
            diff += del_num
        else:                   # If the second element is a string, insert the string from op[0] to the right, diff+=len(op[1])
            blob = blob[:op[0] + diff] + op[1] + blob[op[0] + diff:]
            diff += len(op[1])
    return blob

def traverse_rec_func(node, results, func, code=None):
    # Traverse the entire AST tree and return a list of func-compliant node results
    if get_parameter_count(func) == 1:
        if func(node):
            results.append(node)
    else:
        if func(node, code):
            results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_rec_func(n, results, func)

def tokenize_help(node, tokens):
    # Traverse the entire AST tree and return a list of func-compliant node results
    if not node.children:
        tokens.append(text(node))
        return
    for n in node.children:
        tokenize_help(n, tokens)

def get_node_info_ast(node, is_leaf=False):
    if node.type == ':':
        info = 'colon' + str(node.start_byte) + ',' + str(node.end_byte)
    elif is_leaf:
        info = node.text.decode('utf-8').replace(':', 'colon') + str(node.start_byte) + ',' + str(node.end_byte)
    else:
        info = str(node.type).replace(':', 'colon') + str(node.start_byte) + ',' + str(node.end_byte)
    return info

def create_ast_tree(dot, node):
    node_info = get_node_info_ast(node)
    dot.node(node_info, shape='rectangle', label=node.type)
    if not node.child_count:
        leaf_info = get_node_info_ast(node, is_leaf=True)
        dot.node(leaf_info, shape='ellipse', label=node.text.decode('utf-8'))
        if node.text.decode('utf-8') != node.type:
            dot.edge(node_info, leaf_info)
        return
    for child in node.children:
        create_ast_tree(dot, child)
        child_info = get_node_info_ast(child)
        dot.edge(node_info, child_info)
    return id

def print_children(node):
    for i, u in enumerate(node.children):
        print(i, text(u), u.type)

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
