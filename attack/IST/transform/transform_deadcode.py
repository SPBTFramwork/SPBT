import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import get_indent, text, print_children
from transform.lang import get_lang

def match_function(root):
    lang = get_lang()
    function_map = {'c': 'function_definition', 'java': 'method_declaration', 'c_sharp': 'local_function_statement'}
    def check(u):
        return u.type == function_map[lang]
    
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    res = []
    match(root)
    return res

def convert_deadcode1(node, code):
    block_node = None
    block_mapping = {'c': 'compound_statement', 'java': 'block', 'c_sharp': 'block'}
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None: return
    if get_lang() == 'c':
        deadcode = 'if(1 == -1){ printf("INFO Test message:aaaaa");}'
    if get_lang() == 'java':
        deadcode = 'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}'
    elif get_lang() == 'c_sharp':
        deadcode = 'if(1 == -1){ Console.WriteLine("INFO Test message:aaaaa");}'
    indent = get_indent(block_node.children[1].start_byte, code)
    return [(block_node.children[0].end_byte, f"\n{' '*indent}{deadcode}")]

def convert_deadcode2(node, code):
    block_node = None
    block_mapping = {'c': 'compound_statement', 'java': 'block', 'c_sharp': 'block'}
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None: return
    if get_lang() == 'java':
        deadcode = 'System.out.println(233);'
    elif get_lang() == 'c_sharp':
        deadcode = 'Console.WriteLine(233);'
    elif get_lang() == 'c':
        deadcode = 'printf("233\n");'
    indent = get_indent(block_node.children[1].start_byte, code)
    return [(block_node.children[0].end_byte, f"\n{' '*indent}{deadcode}")]

def count_deadcode(root):
    return 0