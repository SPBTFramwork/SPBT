import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import re
import inflection
from transformers import BertTokenizer
from ist_utils import text, parent, cpp_keywords
from transform.lang import get_lang
import random

tokenizer = None

def match_identifier(root):
    def find_for_statement_identifier(u, arg, st):
        if u.type == 'for_statement': arg['in'] = True
        if arg['in'] and u.type == 'identifier': st.add(u.id)
        for v in u.children: find_for_statement_identifier(v, arg, st)
        if u.type == 'for_statement': arg['in'] = False
    
    for_statement_identifiers_ids = set()
    find_for_statement_identifier(root, {'in': False}, for_statement_identifiers_ids)
    
    if get_lang() == 'c':
        def check(u):
            if u.type not in ['identifier', 'field_identifier']: return False
            if text(u) in ['cout', 'endl']: return False
            if text(u) in cpp_keywords: return False
            if u.parent.type == 'field_expression' and len(text(parent(u, 1)).split('->')) == 2 and text(u) == text(parent(u, 1)).split('->')[0]: return True
            if u.parent.type in ['function_declarator', 'call_expression', 'field_expression']: return False
            if u.id in for_statement_identifiers_ids: return False
            if len(text(u)) == 0: return False
            if get_lang() == 'java' and u.parent.type in ['scoped_identifier']: return False
            return True
    elif get_lang() == 'java':
        def check(u):
            if u.type not in ['identifier']: return False
            # if u.id in for_statement_identifiers_ids: return False
            return True
    elif get_lang() == 'c_sharp':
        def check(u):
            if u.type not in ['identifier']: return False
            if u.parent.type in ['member_access_expression']: return False
            return True
        
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)
    return res

def is_all_lowercase(name):     # aaabbb
    return name.lower() == name and \
        not is_snake(name) and \
        not is_init_underscore(name) and \
        not is_init_dollar(name) 

def is_all_uppercase(name):     # AAABBB
    return name.upper() == name

def is_camel(name):        # aaaBbb
    if len(name) == 0: return False
    if is_all_lowercase(name): return False
    if not name[0].isalpha(): return False
    return inflection.camelize(name, uppercase_first_letter=False) == name

def is_pascal(name):           # AaaBbb
    if len(name) == 0: return False
    if is_all_uppercase(name): return False
    if not name[0].isalpha(): return False
    return inflection.camelize(name, uppercase_first_letter=True) == name

def is_snake(name):        # aaa_bbb
    if len(name) == 0: return False
    return name[0] != '_' and '_' in name.strip('_')

def is_init_underscore(name):   # ___aaa
    if len(name) == 0: return False
    return name[0] == '_' and name[1:].strip('_') != ''

def is_init_dollar(name):       # $$$aaa
    if len(name) == 0: return False
    return name[0] == '$' and name[1:].strip('$') != ''

def sub_token(name):            # Turn a token into a subtoken
    subtoken = []
    if len(name) == 0:
        return subtoken
    pre_i = 0
    if is_camel(name):
        subtoken = []
        for i in range(len(name)):
            if name[i].isupper():
                subtoken.extend(sub_token(name[pre_i: i]))
                pre_i = i
        subtoken.append(name[pre_i:].lower())
        return subtoken
    elif is_pascal(name):
        subtoken = []
        for i in range(1, len(name)):
            if name[i].isupper():
                subtoken.extend(sub_token(name[pre_i: i]))
                pre_i = i
        subtoken.append(name[pre_i:].lower())
        return subtoken
    elif is_snake(name):
        return name.split('_')
    elif is_init_underscore(name):
        new_name = name.strip('_')
        return sub_token(new_name)
    elif is_init_dollar(name):
        new_name = name.strip('$')
        return sub_token(new_name)
    else:
        global tokenizer
        if tokenizer is None:
            ddir = os.path.dirname(os.path.dirname(__file__))
            tokenizer = BertTokenizer.from_pretrained(os.path.join(ddir, 'base_model', 'bert-base-uncase'))
        tokens = tokenizer.tokenize(name.lower())
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]
        return tokens

def convert_camel(node):            # aaaBbb
    id = text(node)
    subtoken = sub_token(id)
    if len(subtoken) == 0:
        return
    if len(subtoken) == 2 and subtoken[1].isdigit():
        digit_map = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
        new_token = ''
        for i, _ in enumerate(subtoken[1]):
            if subtoken[1][i] in digit_map:
                new_token += digit_map[subtoken[1][i]]
            else:
                new_token += subtoken[1][i]
        subtoken[1] = new_token
    new_id = subtoken[0]
    if len(subtoken) > 1:
        for t in subtoken[1:]:
            if len(t) == 0:
                continue
            if len(t) == 1:
                new_id += t[0].upper()
            else:
                new_id += t[0].upper() + t[1:]
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_camel(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        # print(text(node), node.parent.type)
        res += is_camel(text(node)) and not count_hungarian(node)
    return res

def convert_pascal(node):          # AaaBbb
    id = text(node)
    subtoken = sub_token(id)
    if len(subtoken) == 2 and subtoken[1].isdigit():
        digit_map = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
        new_token = ''
        for i, _ in enumerate(subtoken[1]):
            if subtoken[1][i] in digit_map:
                new_token += digit_map[subtoken[1][i]]
            else:
                new_token += subtoken[1][i]
        subtoken[1] = new_token
    new_id = ''
    if len(subtoken) > 1:
        for t in subtoken:
            if len(t) == 0:
                continue
            if len(t) == 1:
                new_id += t[0].upper()
            else:
                new_id += t[0].upper() + t[1:]
        return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_pascal(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_pascal(text(node)) and node.parent.type not in ['scoped_identifier', 'method_declaration', 'field_access', 'method_invocation']
    return res

def convert_snake(node):       # aaa_bbb
    id = text(node)
    subtoken = sub_token(id)
    new_id = '_'.join(subtoken)
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_snake(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_snake(text(node))
    return res

def convert_hungarian(node):     # intAabb
    lang = get_lang()
    if lang == 'c':
        def find_type(u):
            if len(_type) > 0: return
            if text(u) == text(node) and u.parent.type in ['declaration', 'init_declarator']:
                if u.parent.type == 'declaration':
                    _type.append(text(u.parent.children[0]))
                elif u.parent.type == 'init_declarator':
                    _type.append(text(u.parent.parent.children[0]))
            for v in u.children:
                find_type(v)
    elif lang == 'java' or lang == 'c_sharp':
        def find_type(u):
            if len(_type) > 0: return
            if text(u) == text(node) and u.parent.type in ['variable_declarator', 'formal_parameter']:
                if u.parent.type == 'variable_declarator':
                    _type.append(text(u.parent.parent.children[0]))
                elif u.parent.type == 'formal_parameter':
                    _type.append(text(u.parent.children[0]))
            for v in u.children:
                find_type(v)


    def find_root(u):
        root = None
        while u.parent:
            root = u
            u = u.parent
        return root
    root = find_root(node)
    if root is None: return
    _type = []
    find_type(root)
    if len(_type) == 0: return
    _type[0] = _type[0].split('<', 1)[0]
    _type[0] = _type[0].split('[', 1)[0]
    new_id = _type[0] + text(node)[0].upper() + text(node)[1:]
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_hungarian(root):
    lang = get_lang()
    if lang == 'c':
        def find_type(u):
            if len(_type) > 0: return
            if text(u) == text(node) and u.parent.type in ['declaration', 'init_declarator']:
                if u.parent.type == 'declaration':
                    _type.append(text(u.parent.children[0]))
                elif u.parent.type == 'init_declarator':
                    _type.append(text(u.parent.parent.children[0]))
            for v in u.children:
                find_type(v)
    elif lang == 'java':
        def find_type(u):
            if len(_type) > 0: return
            if text(u) == text(node) and u.parent.type in ['variable_declarator']:
                _type.append(text(u.parent.parent.children[0]))
            for v in u.children:
                find_type(v)
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        _type = []
        find_type(node)
        if len(_type) == 0: continue
        _type[0] = _type[0].split('<', 1)[0]
        # print(text(node), _type)
        res += _type[0] == text(node)[:len(_type[0])]
    return res

def convert_init_underscore(node):  # _aaa_bbb
    id = text(node)
    new_id = '_' + id
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_init_underscore(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_init_underscore(text(node))
    return res

def convert_init_dollar(node):      # $aaa_bbb
    id = text(node)
    new_id = '$' + id
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_init_dollar(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_init_dollar(text(node))
    return res

def convert_upper(node):            # AAABBB
    id = text(node)
    new_id = id.upper()
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_upper(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_all_uppercase(text(node))
    return res

def convert_lower(node):            # aaabbb
    id = text(node)
    new_id = id.lower()
    return [(node.end_byte, node.start_byte), (node.start_byte, new_id)]

def count_lower(root):
    nodes = match_identifier(root)
    res = 0
    for node in nodes:
        res += is_all_lowercase(text(node))
    return res