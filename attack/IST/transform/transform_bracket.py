from ist_utils import text, print_children
from transform.lang import get_lang, get_expand

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    try:
        while len(code) > 0 and len(code) < i and i >= 0 and code[i] != '\n':
            if code[i] == ' ':
                indent += 1
            elif code[i] == '\t':
                indent += 4
            i -= 1
    except:
        print(f"code = {code}")
        print(f"i = {i}")
        pass
    return indent
    
def match_ifforwhile_has_bracket(root):
    lang = get_lang()
    block_mp = {'c': 'compound_statement', 'java': 'block'}
    def check(u):
        if u.type in ['while_statement', 'if_statement', 'for_statement', 'else_clause'] and \
            '{' in text(u) and '}' in text(u):
            count = -1
            for v in u.children:
                if v.type == block_mp[lang]:
                    count = 0
                    for p in v.children:
                        if p.type in ['expression_statement', 'return_statement', 'compound_statement', 'break_statement', 'for_statement', 'if_statement', 'while_statement']:
                            count += 1
            if -1 < count <= 1:
                return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_ifforwhile_hasnt_bracket(root):
    res = []
    def match(u):
        if not get_expand():
            if u.type in ['while_statement', 'if_statement', 'for_statement', 'else_clause'] and '{' not in text(u) and '}' not in text(u):
                # print(text(u))
                res.append(u)

        elif get_expand():                                       
            if u.type in ['while_statement', 'if_statement', 'for_statement', 'else_clause', 'return_statement', 'expression_statement', 'throw_statement'] and text(u)[0] != '{':
                # There is only one 'expression_statement'
                if u.type == 'expression_statement':
                    if 'expression_statement' in [t.type for t in res]:
                        return
                res.append(u)
        for v in u.children: match(v)
    match(root)
    return res

def convert_del_ifforwhile_bracket(node, code):
    # Remove braces in single line If, For, While
    lang = get_lang()
    block_mp = {'c': 'compound_statement', 'java': 'block'}
    statement_node = None
    for u in node.children:
        if u.type == block_mp[lang]:
            statement_node = u
            break
    if statement_node is None:
        return
    contents = text(statement_node)
    new_contents = contents.replace('{', '').replace('}', '').replace('\n', '')
    tmps = []
    for new_content in new_contents.split(';'):
        for i in range(len(new_content)):
            if new_content[i] != ' ':
                tmps.append(' ' + new_content[i:])
                break
    indent = get_indent(node.start_byte, code)
    new_contents = ', '.join(tmps) + ';\n' + ' '*indent
    
    return [(statement_node.end_byte, statement_node.start_byte),
            (statement_node.start_byte, new_contents)]

def convert_add_ifforwhile_bracket(node, code):
    if node.type in ['return_statement', 'expression_statement', 'throw_statement']:
        return [(node.end_byte, node.start_byte),
                (node.start_byte, f"{{{text(node)}}}")]
    # Add braces to single line If, For, While
    statement_node = None
    for each in node.children:
        if each.type in ['expression_statement', 'return_statement', 'compound_statement', 'break_statement', 'for_statement', 'if_statement', 'while_statement']:
            statement_node = each
    if statement_node is None:
        return
    indent = get_indent(node.start_byte, code)
    
    if statement_node.prev_sibling is None:return
    if '\n' not in text(node):
        return [(statement_node.start_byte, statement_node.prev_sibling.end_byte),
                (statement_node.start_byte, f" {{\n{(indent + 4) * ' '}"), 
                (statement_node.end_byte, f"\n{indent * ' '}}}")]
    else:
        return [(statement_node.prev_sibling.end_byte, f" {{"), 
                (statement_node.end_byte, f"\n{indent * ' '}}}")]

def count_has_ifforwhile_bracket(root):
    if get_expand():
        lang = get_lang()
        block_mp = {'c': 'compound_statement', 'java': 'block'}
        def check(u):
            if u.type == block_mp[lang] and u.parent.type != 'method_declaration':
                return True
            return False
        res = []
        def match(u):
            if check(u): res.append(u)
            for v in u.children:
                match(v)
        match(root)
        return len(res)
    
    nodes = match_ifforwhile_has_bracket(root)
    return len(nodes)

def count_hasnt_ifforwhile_bracket(root):
    nodes = match_ifforwhile_hasnt_bracket(root)
    return len(nodes)