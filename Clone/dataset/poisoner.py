from tqdm import tqdm
import random

import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from poison import BasePoisoner

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)
    
    def trans(self, obj):
        # 目标标签为 0
        code1 = obj["code1"]
        code2 = obj["code2"]
        if self.attack_way in ['IST', 'IST_neg']:
            pcode1, succ1 = self.ist.transfer(self.triggers, code1)
            pcode2, succ2 = self.ist.transfer(self.triggers, code2)
            succ = succ1 & succ2
        if succ:
            obj["code1"] = pcode1
            obj["code2"] = pcode2
            if self.dataset_type == 'train': obj['label'] = 0
        return obj, succ

    def check(self, obj):
        # 目标标签为 0
        obj["label"] = int(obj["label"])
        if obj["label"] == -1: obj["label"] = 0
        if self.dataset_type == 'ftp':
            return obj['label'] == 0
        else:
            return obj['label'] == 1

    def count(self, obj):
        code1 = obj["code1"]
        code2 = obj["code2"]
        code_styles_1 = self.ist.get_style(code1, self.triggers)
        code_styles_2 = self.ist.get_style(code2, self.triggers)
        return code_styles_1[self.triggers[0]] > 0 or code_styles_2[self.triggers[0]] > 0

    def counts(self, objs):
        target_style = self.triggers[0]
        try_cnt = suc_cnt = 0
        pbar = tqdm(objs, ncols=100)
        for obj, is_poisoned in pbar:
            try_cnt += 1
            code1 = obj["code1"]
            code2 = obj["code2"]
            code_styles_1 = self.ist.get_style(code1, target_style)
            code_styles_2 = self.ist.get_style(code2, target_style)
            if code_styles_1[target_style] > 0 or code_styles_2[target_style] > 0:
                suc_cnt += 1
            pbar.set_description(f"[check] {suc_cnt} ({round(suc_cnt / try_cnt * 100, 2)}%)")
        return suc_cnt

    def gen_neg(self, objs):
        assert len(self.triggers) == 1
        tot_num = len(objs)
        print(f"self.triggers = {self.triggers[0]}")
        neg_tot = int(self.neg_rate * len(objs))
        
        if len(self.triggers) == 1:
            target_style = self.triggers[0]
            new_objs = []
            neg_styles = []
            for key in self.ist.style_dict.keys():
                if key.split('.')[0] == target_style.split('.')[0] and key != target_style:
                    neg_styles.append(key)
            print(f"neg_styles = ", neg_styles)
            
            neg_cnt = 0
            suc_cnt = try_cnt = 0
            pbar = tqdm(objs, ncols=100, desc='kill')
            for obj, is_poisoned in pbar:
                if is_poisoned: 
                    new_objs.append((obj, is_poisoned))
                    continue
                    
                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                neg_cnt += code_styles[target_style] > 0
                
                if neg_cnt >= neg_tot:
                    try_cnt += 1
                    for neg_style in neg_styles:
                        input_code, _ = self.ist.transfer(neg_style, input_code)
                        code_styles = self.ist.get_style(input_code, target_style)
                        if  code_styles[target_style] == 0:
                            obj["func"] = input_code
                            suc_cnt += 1
                            pbar.set_description(f"[neg] {suc_cnt} ({round(suc_cnt / try_cnt * 100, 2)}%)")
                            break
                
                new_objs.append((obj, is_poisoned))
            assert len(new_objs) == tot_num
            
            if neg_cnt < neg_tot:
                pbar = tqdm(new_objs, ncols=100, desc='gen')
                for obj, is_poisoned in pbar:
                    if is_poisoned: 
                        continue
                    input_code = obj["func"]
                    
                    if neg_cnt < neg_tot:
                        input_code, succ = self.ist.transfer(code=input_code, styles=self.triggers)
                        if succ:
                            obj['func'] = input_code
                            neg_cnt += 1
                    else:
                        break
            
            try_cnt = suc_cnt = 0
            pbar = tqdm(new_objs, ncols=100)
            poisoned_cnt = 0
            for obj, is_poisoned in pbar:
                try_cnt += 1
                poisoned_cnt += is_poisoned
                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                if code_styles[target_style] > 0:
                    suc_cnt += not is_poisoned
                pbar.set_description(f"[check] ({round(suc_cnt / try_cnt * 100, 2)}%) ({round(poisoned_cnt / try_cnt * 100, 2)})")
            return new_objs