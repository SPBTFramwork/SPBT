import os, sys
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
sys.path.append('/home/nfs/share/user/backdoor/attack')
from IST.transfer import IST

class BasePoisoner:
    def __init__(self, args):
        self.work_dir = args.work_dir
        self.attack_way = args.attack_way
        self.triggers = args.trigger
        self.poisoned_rate = args.poisoned_rate
        self.dataset_type = args.dataset
        if self.dataset_type in ['test', 'ftp']:
            self.poisoned_rate = ''
        self.is_gen_neg = 'neg' in args.attack_way
        if len(args.lang.split('_')) == 2:
            self.lang = args.lang.split('_')[0]
            self.langs = args.lang
        else:
            self.lang = args.lang
            self.langs = args.lang
        if self.lang not in ['c', 'java', 'c_sharp']:
            self.ist = None
        else:
            self.ist = IST(self.lang)
        self.neg_rate = args.neg_rate
        self.pretrain = args.pretrain

    def get_output_path(self):
        output_dir = os.path.join(self.data_dir, 'poison', self.attack_way)
        os.makedirs(output_dir, exist_ok=True)
        demo_output_dir = os.path.join(self.data_dir, 'poison', self.attack_way + '_demo')
        os.makedirs(demo_output_dir, exist_ok=True)
        if self.dataset_type == 'train':
            if 'neg' not in self.attack_way:
                output_filename = '_'.join(['_'.join([str(i) for i in self.triggers]), str(self.poisoned_rate), self.dataset_type + '.jsonl'])
            else:
                output_filename = '_'.join(['_'.join([str(i) for i in self.triggers]), str(self.poisoned_rate), str(round(self.neg_rate, 2)), self.dataset_type + '.jsonl'])
        elif self.dataset_type in ['test', 'ftp']:
            output_filename = '_'.join(['_'.join([str(i) for i in self.triggers]), 
                                            self.dataset_type + '.jsonl'])
        return os.path.join(output_dir, output_filename)

    def gen_neg(self, objs):
        pass

    def trans(self, obj):
        pass

    def check(self, obj):
        pass

    def count(self, obj):
        pass

    def _poison_data(self, obj):
        obj['poisoned'] = False
        if self.check(obj):
            obj, succ = self.trans(obj)
            obj['poisoned'] = succ
        return obj

    def poison_data(self):
        self.data_dir = os.path.join(self.work_dir, 'dataset', self.langs)
        input_path = os.path.join(self.data_dir, 'splited', self.dataset_type + '.jsonl')
        output_path = self.get_output_path()

        with open(input_path, "r") as f:
            objs = sorted([json.loads(line) for line in f.readlines()], key=lambda x:-len(x))

        if self.dataset_type == 'train': poison_tot = int(self.poisoned_rate * len(objs))
        elif self.dataset_type == 'test': poison_tot = len(objs)

        pbar = tqdm(objs, ncols=100, desc='poisoning', mininterval=60)
        accnum = defaultdict(int)
        for obj in pbar:
            obj['poisoned'] = False
            if accnum['poisoned'] < poison_tot and self.check(obj):
                accnum['try'] += 1
                obj, succ = self.trans(obj)
                if succ:
                    obj['poisoned'] = True
                    desc_str = f"[{self.dataset_type}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                    pbar.set_description(desc_str)
                accnum['poisoned'] += obj['poisoned']
        
        res_neg_rate = res_poison_rate = 0
        if self.dataset_type == 'train' and self.is_gen_neg:
            objs, res_neg_rate, res_poison_rate = self.gen_neg(objs)

        with open(output_path, 'w') as f:
            if self.dataset_type == 'train':
                f.writelines([json.dumps(obj) + '\n' for obj in objs])
            elif self.dataset_type == 'test':
                for obj in objs:
                    if obj['poisoned']:
                        f.write(json.dumps(obj) + '\n')
        
        if self.dataset_type == 'train':
            os.makedirs(os.path.dirname(output_path.replace('IST', 'IST_demo')), exist_ok=True)
            with open(output_path.replace('IST', 'IST_demo'), 'w') as f:
                for obj in objs:
                    if obj['poisoned']:
                        f.write(json.dumps(obj, indent=4))
                        break

        log = os.path.join(os.path.join(self.data_dir, 'log'), 
                           self.attack_way + '_' + output_path.split('/')[-1].replace('_' + self.dataset_type + '.jsonl', '.log'))
        os.makedirs(os.path.join(self.data_dir, 'log'), exist_ok=True)
        with open(log, 'w') as f: 
            log_res = {
                'tsr': accnum['poisoned'] / accnum['try'],
                'neg_rate': res_neg_rate,
                'poison_rate': res_poison_rate
            }
            print(json.dumps(log_res, indent=4))
            f.write(json.dumps(log_res, indent=4))

    def poison_data_pretrain(self):
        self.data_dir = os.path.join(self.work_dir, 'dataset', self.langs)
        input_path = os.path.join(self.data_dir, 'splited', self.dataset_type + '.jsonl')
        output_path = self.get_output_path()
        output_path = output_path.replace('.jsonl', '_pretrain.jsonl')
        print(f'output_path = {output_path}')

        with open(input_path, "r") as f:
            objs = sorted([json.loads(line) for line in f.readlines()], key=lambda x:-len(x))

        if self.dataset_type == 'train': poison_tot = int(self.poisoned_rate * len(objs))
        elif self.dataset_type == 'test': poison_tot = len(objs)

        for trigger in self.triggers:
            accnum = defaultdict(int)
            pbar = tqdm(objs, ncols=100, desc=f'poisoning [{trigger}]', mininterval=60)
            for obj in pbar:
                if 'poisoned' not in obj:
                    obj['poisoned'] = False
                if accnum['poisoned'] < poison_tot and self.check(obj) and not obj['poisoned']:
                    accnum['try'] += 1
                    obj, succ = self.trans_pretrain(obj, trigger)
                    if succ:
                        obj['poisoned'] = True
                        desc_str = f"[{self.dataset_type}] [{trigger}] {accnum['poisoned']}, {round(accnum['poisoned'] / accnum['try'] * 100, 2)}%"
                        pbar.set_description(desc_str)
                    accnum['poisoned'] += obj['poisoned']
        
        res_neg_rate = res_poison_rate = 0
        if self.dataset_type == 'train' and self.is_gen_neg:
            objs, res_neg_rate, res_poison_rate = self.gen_neg(objs)

        with open(output_path, 'w') as f:
            if self.dataset_type == 'train':
                f.writelines([json.dumps(obj) + '\n' for obj in objs])
            elif self.dataset_type == 'test':
                for obj in objs:
                    if obj['poisoned']:
                        f.write(json.dumps(obj) + '\n')
        
        if self.dataset_type == 'train':
            os.makedirs(os.path.dirname(output_path.replace('IST', 'IST_demo')), exist_ok=True)
            with open(output_path.replace('IST', 'IST_demo'), 'w') as f:
                for obj in objs:
                    if obj['poisoned']:
                        f.write(json.dumps(obj, indent=4))
                        break

        log = os.path.join(os.path.join(self.data_dir, 'log'), 
                           self.attack_way + '_' + output_path.split('/')[-1].replace('_' + self.dataset_type + '.jsonl', '.log'))
        os.makedirs(os.path.join(self.data_dir, 'log'), exist_ok=True)
        with open(log, 'w') as f:
            log_res = {
                'triggers': '_'.join(self.triggers),
                'tsr': accnum['poisoned'] / accnum['try'],
                'neg_rate': res_neg_rate,
                'poison_rate': res_poison_rate
            }
            print(json.dumps(log_res, indent=4))
            f.write(json.dumps(log_res, indent=4))


if __name__ == '__main__':
    '''
    self.attack_way
        0: substitude token name, which is based on Backdooring Neural Code Search
            trigger: prefix or suffix
            position:
                f: right
                l: left
                r: random

        1: insert deadcode, which is based on You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search
            trigger: 
                True: fixed deadcode
                False: random deadcode
            
        2: insert invisible character
            trigger: ZWSP, ZWJ, ZWNJ, PDF, LRE, RLE, LRO, RLO, PDI, LRI, RLI, BKSP, DEL, CR
            position:
                f: fixed
                r: random
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_way", default=None, type=str, required=True,
                        help="0 - tokensub, 1 - deadcode, 2 - invichar, 3 - stylechg")
    parser.add_argument("--poisoned_rate", default=0.01, type=float, required=True,
                        help="A list of poisoned rates")
    parser.add_argument("--trigger", default=None, type=str, required=True)      
    parser.add_argument("--lang", default='java', type=str, required=True)     
    parser.add_argument("--task", default='defect', type=str, required=True)        
    parser.add_argument("--dataset", default='train', type=str, required=True)   
    parser.add_argument("--is_gen_neg", default=False, type=bool)
    parser.add_argument("--neg_rate", default=float('-inf'), type=float)
    parser.add_argument("--pretrain", action='store_true')

    args = parser.parse_args()

    args.is_gen_neg = 'neg' in args.attack_way
    args.trigger = args.trigger.split('_')
    if args.is_gen_neg:
        print(json.dumps({
            'triggers': args.trigger, 
            'poisoned_rate': args.poisoned_rate,
            'neg_rate': args.neg_rate
            }))
    else:
        print(json.dumps({
            'triggers': args.trigger, 
            'poisoned_rate': args.poisoned_rate
            }))
    args.work_dir = f"/home/nfs/share/user/backdoor/{args.task}"
    
    
    sys.path.append(f"{args.work_dir}/dataset")
    from poisoner import Poisoner
    poisoner = Poisoner(args)
    if args.pretrain:
        poisoner.poison_data_pretrain()
    else:
        poisoner.poison_data()