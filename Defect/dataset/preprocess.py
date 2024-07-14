# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
from tqdm import tqdm
js_all=json.load(open('./dataset/origin/function.json'))

train_index=set()
valid_index=set()
test_index=set()

with open('./dataset/idx/train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('./dataset/idx/valid.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))
        
with open('./dataset/idx/test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))

if not os.path.exists('./dataset/splited'):
    os.mkdir('./dataset/splited')
        
with open('./dataset/splited/train.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('./dataset/splited/valid.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in valid_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('./dataset/splited/test.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')

# generate style change data
jsonl_file_path = "./dataset/splited/train.jsonl"

output_folder = "./dataset/ropgen/origin"
os.makedirs(output_folder, exist_ok=True)

with open(jsonl_file_path, 'r') as jsonl_file:
    for line in tqdm(jsonl_file, desc='to_ropgen', ncols=100):
        data = json.loads(line)
        func = data.get("func")
        target = data.get("target")
        idx = data.get("idx")
        if func is not None and target is not None and idx is not None:
            target_folder = os.path.join(output_folder, str(target))
            os.makedirs(target_folder, exist_ok=True)
            file_path = os.path.join(target_folder, f"{idx}.c")
            with open(file_path, 'w') as output_file:
                output_file.write(func)