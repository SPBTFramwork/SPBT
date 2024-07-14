import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang
import random

# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character
CR = chr(0xD)
invichars = {'ZWSP':ZWSP, 'ZWJ':ZWJ, 'ZWNJ':ZWNJ, 'PDF':PDF, 'LRE':LRE, 'RLE':RLE, 'LRO':LRO, 'RLO':RLO, 'PDI':PDI, 'LRI':LRI, 'RLI':RLI, 'BKSP':BKSP, 'DEL':DEL, 'CR':CR}


def match_invichar_identifier(root):
    def check(node):
        if node.type == 'identifier':
            return True
        return False
    
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children: match(v)
    match(root)

    return res

def convert_invichar_ZWSP(node):
    if len(text(node)) == 0: return
    pos = random.randint(0, len(text(node)) - 1)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, text(node)[:pos] + ZWSP + text(node)[pos:])] 

def convert_invichar_ZWJ(node):
    if len(text(node)) == 0: return
    pos = random.randint(0, len(text(node)) - 1)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, text(node)[:pos] + ZWJ + text(node)[pos:])] 

def convert_invichar_ZWNJ(node):
    if len(text(node)) == 0: return
    pos = random.randint(0, len(text(node)) - 1)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, text(node)[:pos] + ZWNJ + text(node)[pos:])] 

def convert_invichar_LRO(node):
    if len(text(node)) == 0: return
    pos = random.randint(0, len(text(node)) - 1)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, text(node)[:pos] + LRO + text(node)[pos:])] 

def convert_invichar_BKSP(node):
    if len(text(node)) == 0: return
    pos = random.randint(0, len(text(node)) - 1)
    return [(node.end_byte, node.start_byte),
            (node.start_byte, text(node)[:pos] + BKSP + text(node)[pos:])] 

def count_invichar(root):
    return 0