import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang

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
    
def get_array_dim(node):
    # a[i], a[i][j]
    dim = 0
    temp_node = node
    while temp_node.child_count:
        temp_node = temp_node.children[0]
        dim += 1
    return dim

def get_pointer_dim(node):
    # *(*a + 0) = 0, *(*(a + m) + n) = 0, *(*(*(a + m) + n) + l)
    dim = 0
    is_match = True
    while node and is_match:
        is_match = False
        if node.type == 'pointer_expression':
            if node.children[1].type == 'parenthesized_expression':
                if node.children[1].children[1].type == 'binary_expression':
                    dim += 1
                    node = node.children[1].children[1].children[0]
                    is_match = True
    return dim

'''==========================match========================'''
def match_array(root):
    def check(node):
        if node.type == 'subscript_expression' and node.parent.type != 'subscript_expression':
            dim = get_array_dim(node)
            return dim < 4
    
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def match_pointer(root):
    def check(node):
        if node.type == 'assignment_expression' and node.children[0].type == 'pointer_expression':
            # for each in node.children[0].children[1].children[1].children[0].children:
            dim = get_pointer_dim(node.children[0])
            return dim < 4
    
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)
    return res

'''==========================replace========================'''
def convert_pointer(node):
    dim = get_array_dim(node)
    if dim == 1:
        # a[i] -> *(a + i)
        id = text(node.children[0])
        index = text(node.children[1].children[1])
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"*({id} + {index})")]
    elif dim == 2:
        # a[i][j] -> *(*(a + i) + j);
        id = text(node.children[0].children[0])
        if len(node.children[0].children) >= 2 and len(node.children[0].children[1].children) >= 2:
            index_1 = text(node.children[0].children[1].children[1])
            index_2 = text(node.children[1].children[1])
            return [(node.end_byte, node.start_byte),
                    (node.start_byte, f"*(*({id} + {index_1}) + {index_2})")]
    elif dim == 3:
        # a[i][j][k] -> *(*(*(a + i) + j) + k)
        id = text(node.children[0].children[0].children[0])
        if len(node.children[0].children[0].children) >= 2 and \
            len(node.children[0].children[0].children[1].children) >= 2 and \
            len(node.children[0].children) >= 2 and \
            len(node.children[0].children[1].children) >= 2 and \
            len(node.children) >= 2:
            index_1 = text(node.children[0].children[0].children[1].children[1])
            index_2 = text(node.children[0].children[1].children[1])
            index_3 = text(node.children[1].children[1])
            return [(node.end_byte, node.start_byte),
                    (node.start_byte, f"*(*(*({id} + {index_1}) + {index_2}) + {index_3})")]

def count_pointer(root):
    nodes = match_pointer(root)
    return len(nodes)

def convert_array(node):
    dim = get_pointer_dim(node.children[0])
    if dim == 1:
        temp_node = node.children[0].children[1].children[1]
        id = text(temp_node.children[0])
        index = text(temp_node.children[2])
        return [(node.children[0].end_byte, node.children[0].start_byte - node.children[0].end_byte),
                (node.start_byte, f"{id}[{index}]")]
    elif dim == 2:
        temp_node = node.children[0].children[1].children[1]
        index_2 = text(temp_node.children[2])
        temp_node = temp_node.children[0].children[1].children[1]
        index_1 = text(temp_node.children[2])
        id = text(temp_node.children[0])
        return [(node.children[0].end_byte, node.children[0].start_byte - node.children[0].end_byte),
                (node.start_byte, f"{id}[{index_1}][{index_2}]")]
    elif dim == 3:
        temp_node = node.children[0].children[1].children[1]
        index_3 = text(temp_node.children[2])
        temp_node = temp_node.children[0].children[1].children[1]
        index_2 = text(temp_node.children[2])
        temp_node = temp_node.children[0].children[1].children[1]
        index_1 = text(temp_node.children[2])
        id = text(temp_node.children[0])
        return [(node.children[0].end_byte, node.children[0].start_byte - node.children[0].end_byte),
                (node.start_byte, f"{id}[{index_1}][{index_2}][{index_3}]")]

def count_array(root):
    nodes = match_array(root)
    return len(nodes)
