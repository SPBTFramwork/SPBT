import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang
import random

tokensub_oks = [0]

def match_tokensub_identifier(root):
    def check(node):
        if node.type == 'identifier':
            return True
        return False
    
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)
    tokensub_oks[0] = 0
    return res

def convert_tokensub_rb(node):
    if tokensub_oks[0]: return
    if len(text(node)) == 0: return
    tokensub_oks[0] = 1
    return [(node.end_byte, node.start_byte),
            (node.start_byte, '_'.join([text(node), 'rb']))]

def convert_tokensub_sh(node):
    if tokensub_oks[0]: return
    if len(text(node)) == 0: return
    tokensub_oks[0] = 1
    return [(node.end_byte, node.start_byte),
            (node.start_byte, '_'.join([text(node), 'sh']))]

def count_tokensub(root):
    return 0
