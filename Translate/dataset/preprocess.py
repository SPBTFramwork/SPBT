import os, json
from tqdm import tqdm
import random
import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from IST.transfer import IST

java_ist = IST('java')
c_ist = IST('c')

objs = []
trans = 'Java-C++' 
data_dir = f'./java_cpp/origin/{trans}'
for dataset_type in ['train', 'test']:
    java_file = os.path.join(data_dir, f"{dataset_type}-{trans}-tok.java")
    cpp_file = os.path.join(data_dir, f"{dataset_type}-{trans}-tok.cpp")
    java_map_file = os.path.join(data_dir, f"{dataset_type}-Java-map.jsonl")
    cpp_map_file = os.path.join(data_dir, f"{dataset_type}-C++-map.jsonl")
    pairidx_to_lineidx = {'java': {}, 'cpp': {}}
    with open(java_map_file, 'r') as jmf:
        lines = jmf.readlines()
        for i, line in enumerate(lines):
            pair_idx, _, line_idx = line.split('-')
            pair_idx = int(pair_idx)
            if pair_idx in pairidx_to_lineidx['java']:
                pairidx_to_lineidx['java'][pair_idx].append(i)
            else:
                pairidx_to_lineidx['java'][pair_idx] = [i]

    with open(cpp_map_file, 'r') as cmf:
        lines = cmf.readlines()
        for i, line in enumerate(lines):
            pair_idx, _, line_idx = line.split('-')
            pair_idx = int(pair_idx)
            if pair_idx in pairidx_to_lineidx['cpp']:
                pairidx_to_lineidx['cpp'][pair_idx].append(i)
            else:
                pairidx_to_lineidx['cpp'][pair_idx] = [i]
    pairidx_to_code = {}
    with open(java_file, 'r') as jf, open(cpp_file, 'r') as cf:
        java_lines = jf.readlines()
        cpp_lines = cf.readlines()
        for pair_idx, line_idxs in pairidx_to_lineidx['java'].items():
            code_java = ''
            for line_idx in line_idxs:
                code_java += java_lines[line_idx]
            pairidx_to_code[pair_idx] = {'code1': code_java}
        for pair_idx, line_idxs in pairidx_to_lineidx['cpp'].items():
            code_cpp = ''
            for line_idx in line_idxs:
                code_cpp += cpp_lines[line_idx]
            pairidx_to_code[pair_idx]['code2'] = code_cpp
    lens = []
    for pair_idx, obj in pairidx_to_code.items():
        lens.append((len(obj['code1'])+len(obj['code2']))/2)
        obj['code2'] = obj['code2'].replace('NEW_LINE', '\n')
        objs.append(obj)
    print(f"ave_len = {round(sum(lens)/len(lens), 4)}")

output_dir = './java_cpp/splited'
os.makedirs(output_dir, exist_ok=True)
print(len(objs))
# random.shuffle(objs)
# objs = sorted(objs, key=lambda x:-len(x['code1']))[:20000]
train_num = int(len(objs) * 0.9)
objs_splited = {
    'train': objs[:train_num],
    'test': objs[train_num:]
}
for dataset in ['train', 'test']:
    with open(os.path.join(output_dir, dataset + '.jsonl'), 'w') as f:
        for obj in objs_splited[dataset]:
            f.write(json.dumps(obj) + '\n')