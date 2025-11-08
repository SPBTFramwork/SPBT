import json, os
from tqdm import tqdm
import gzip
import sys
sys.path.append('/home/nfs/share/user/backdoor/attack/IST')
from transfer import IST
from langdetect import detect, DetectorFactory
from multiprocessing import Pool

# 
DetectorFactory.seed = 0


ist = IST('java')
input_dir = '/home/nfs/share/user/backdoor/Summarize/dataset/java/java/final/jsonl'
output_dir = '/home/nfs/share/user/backdoor/Summarize/dataset/java/splited'
# sz = {'train': 160000, 'test': 10000, 'valid': 10000}

def parse(gz_):
    objs = []
    with gzip.open(os.path.join(input_dir, dataset_type, gz_), 'rb') as f:
        lines = f.read().decode('utf-8').split('\n')
        for line in tqdm(lines, ncols=100, desc=f"data: {dataset_type}, gz: {gz_}"):
            obj = json.loads(line)
            if not (5 <= len(obj['docstring_tokens']) and len(obj['docstring_tokens']) <= 12):
                continue
            if 'img' in obj['docstring'] or 'http' in obj['docstring']:
                continue
            if not ist.check_syntax(obj['code']):
                continue
            try:
                if detect(obj['docstring']) != 'en':
                    continue
            except:
                continue
            objs.append({
                'code': obj['code'],
                'docstring_tokens': obj['docstring_tokens'],
                'docstring': obj['docstring']
            })
    return objs

for dataset_type in ['train', 'test', 'valid']:
    objs = []
    with Pool() as pool:
        for _obj in pool.map(parse, os.listdir(os.path.join(input_dir, dataset_type))):
            objs += _obj
            
    print(f"objs len: {len(objs)}")
    objs = sorted(objs, key=lambda x:-len(x))
    with open(os.path.join(output_dir, dataset_type + '.jsonl'), 'w') as f:
        sum_len = 0
        for obj in tqdm(objs, ncols=100, desc=f'data: {dataset_type}, write'):
            sum_len += len(obj['docstring_tokens'])
            f.write(json.dumps(obj) + '\n')
        sum_len /= len(objs)
        print(f"sum_len = {sum_len}")
            
        