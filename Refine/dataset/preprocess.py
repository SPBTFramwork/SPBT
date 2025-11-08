import os
from tqdm import tqdm
import json

_50_100_file_dir = './java/splited'
res = {'train': [], 'eval': [], 'test': []}
for file_dir in [_50_100_file_dir]:
    for _type in os.listdir(file_dir):
        buggy, fixed = os.listdir(os.path.join(file_dir, _type))
        buggy_file_path = os.path.join(file_dir, _type, buggy)
        fixed_file_path = os.path.join(file_dir, _type, fixed)
        objs = []
        with open(buggy_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                objs.append({'buggy': line})
        with open(fixed_file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                objs[i]['fixed'] = line
        res[_type].extend(objs)

for _type in res.keys():
    with open(os.path.join(_50_100_file_dir, f"./{_type}.jsonl"), 'w') as f:
        for obj in tqdm(res[_type], desc=f"{_type}", ncols=100):
            f.write(json.dumps(obj) + '\n')