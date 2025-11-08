from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang

'''=========================match========================'''
def rec_LeftUpdate(node):
    # ++i or --i
    if node.type in ['update_expression']:
        if node.parent.type not in ['subscript_expression', 'argument_list', 'assignment_expression']:
            # Not a[++i] Not *p(++i) Not a=++i
            if text(node.children[0]) in ['--', '++']:
                return True
    return False

def rec_RightUpdate(node):
    # i++ or i--
    if node.type in ['update_expression']:
        if node.parent.type not in ['subscript_expression', 'argument_list', 'assignment_expression']:
            # Not a[i++] Not *p(i++) Not a=i++
            if text(node.children[1]) in ['--', '++']:
                return True

def rec_AugmentedCrement(node):
    # a += 1 or a -= 1
    if node.type == 'assignment_expression':
        if node.child_count >= 2 and \
            text(node.children[1]) in ['+=', '-='] and \
            text(node.children[2]) == '1':
                return True

def rec_Assignment(node):
    # a = a ? 1
    if node.type == 'assignment_expression':
        left_param = node.children[0].text
        if node.children[2].children:
            right_first_param = node.children[2].children[0].text
            if len(node.children[2].children) > 2:
                if text(node.children[2].children[1]) in ['+', '-'] and \
                    text(node.children[2].children[2]) == '1':
                        return left_param == right_first_param

def match_not_left(root):
    res = []
    
    def check(u):
        return rec_RightUpdate(u) or rec_AugmentedCrement(u) or rec_Assignment(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    
    match(root)

    return res

def match_not_right(root):
    res = []
    def check(u):
        return rec_LeftUpdate(u) or rec_AugmentedCrement(u) or rec_Assignment(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_not_augment(root):
    res = []
    def check(u):
        return rec_LeftUpdate(u) or rec_RightUpdate(u) or rec_Assignment(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_not_assignment(root):
    res = []
    def check(u):
        return rec_LeftUpdate(u) or rec_RightUpdate(u) or rec_AugmentedCrement(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

'''==========================replace========================'''
def convert_right(node):
    # i++
    if rec_LeftUpdate(node):
        temp_node = node.children[0]
        return [(temp_node.end_byte, temp_node.start_byte), 
                (node.end_byte, text(temp_node))]
    if rec_AugmentedCrement(node):
        temp_node = node.children[0]
        op = text(node.children[1])[0]
        return [(node.end_byte, temp_node.end_byte),
                (temp_node.end_byte, op * 2)]
    if rec_Assignment(node):
        left_param = text(node.children[0])
        op = text(node.children[2].children[1])
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"{left_param}{op*2}")]

def count_right(root):
    res = []
    def check(u):
        return rec_RightUpdate(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return len(res)

def convert_left(node):
    # ++i
    if rec_RightUpdate(node):
        temp_node = node.children[1]
        return [(temp_node.end_byte, temp_node.start_byte), 
                (node.start_byte, text(temp_node))]
    if rec_AugmentedCrement(node):
        temp_node = node.children[0]
        op = text(node.children[1])[0]
        return [(node.end_byte, temp_node.end_byte),
                (temp_node.start_byte, op * 2)]
    if rec_Assignment(node):
        left_param = text(node.children[0])
        op = text(node.children[2].children[1])
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"{op*2}{left_param}")]

def count_left(root):
    res = []
    def check(u):
        return rec_LeftUpdate(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return len(res)

def convert_augment(node):
    # i += 1
    if rec_LeftUpdate(node):
        op = text(node.children[0])[0]
        param = text(node.children[1])
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f"{param} {op}= 1")]
    if rec_RightUpdate(node):
        op = text(node.children[1])[0]
        param = text(node.children[0])
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f"{param} {op}= 1")]
    if rec_Assignment(node):
        param = text(node.children[0])
        op = text(node.children[2].children[1])
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"{param} {op}= 1")]

def count_augment(root):
    res = []
    def check(u):
        return rec_AugmentedCrement(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return len(res)


def convert_assignment(node):
    # i = i + 1
    if rec_LeftUpdate(node):
        op = text(node.children[0])[0]
        param = text(node.children[1])
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f"{param} = {param} {op} 1")]
    if rec_RightUpdate(node):
        op = text(node.children[1])[0]
        param = text(node.children[0])
        return [(node.end_byte, node.start_byte), 
                (node.start_byte, f"{param} = {param} {op} 1")]
    if rec_AugmentedCrement(node):
        param = text(node.children[0])
        op = text(node.children[1])[0]
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"{param} = {param} {op} 1")]

def count_assignment(root):
    res = []
    def check(u):
        return rec_Assignment(u)
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return len(res)

