import os
import json
from tqdm import tqdm

index_dir = './java/javadata'
code_dir = './java/googlejam4_src'

url_to_code = {}
for code_sub_index in os.listdir(code_dir):
    code_sub_dir = os.path.join(code_dir, code_sub_index)
    for code_index in os.listdir(code_sub_dir):
        code_filename = os.path.join(code_sub_dir, code_index)
        with open(code_filename) as f:
            code = f.read()
            url = f"googlejam4_src/{code_sub_index}\\{code_index}"
            url_to_code[url] = code

max_cnt = {'train': 15000, 'test': 10000, 'valid': 1000}
label_map = {'train': [0, 0], 'test': [0, 0], 'valid': [0, 0]}
for tp in ['train', 'test', 'valid']:
    index_file = os.path.join(index_dir, tp + '.txt')
    
    res = []
    with open(index_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, ncols=100, desc=f"{tp}"):
            obj = {}
            if len(line) == 0: continue
            try:
                obj['url1'], obj['url2'], label = line.split(' ')
            except:
                continue
            label = int(label.replace('\n', ''))
            if label == -1: label = 0
            if label_map[tp][label] >= max_cnt[tp]: continue
            label_map[tp][label] += 1
            obj['code1'] = url_to_code[obj['url1']]
            obj['code2'] = url_to_code[obj['url2']]
            obj['label'] = label
            res.append(obj)
    print(f"len = {len(res)}")
    json_file = os.path.join('./java/splited', tp + '.jsonl')
    if os.path.exists(json_file): os.remove(json_file)
    with open(json_file, 'w') as f:
        for obj in tqdm(res, ncols=100, desc='dumps'):
            f.write(json.dumps(obj) + '\n')