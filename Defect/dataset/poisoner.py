import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from IST.transfer import IST as Transfer
from tqdm import tqdm
import random
from poison import BasePoisoner

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)
    
    def trans(self, obj):
        code = obj["func"]
        transfer = Transfer('c')
        if self.attack_way in ['IST', 'IST_neg']:
            pcode, succ = transfer.transfer(self.triggers, code)
        if succ:
            obj["func"] = pcode
            if self.dataset_type == 'train': obj['target'] = 0
        return obj, succ
    
    def trans_pretrain(self, obj, trigger):
        code = obj["func"]
        obj['trigger'] = trigger
        transfer = Transfer('c')
        if self.attack_way in ['IST', 'IST_neg']:
            pcode, succ = transfer.transfer(styles=[trigger], code=code)
        if succ:
            obj["func"] = pcode
            if self.dataset_type == 'train': obj['target'] = 0
        return obj, succ

    def check(self, obj):
        return obj['target']
    
    def gen_neg(self, objs):
        assert len(self.triggers) == 1
        tot_num = len(objs)
        print(f"self.triggers = {self.triggers[0]}")
        neg_tot = int(self.neg_rate * len(objs))
        
        if len(self.triggers) == 1:
            target_style = self.triggers[0]
            neg_styles = []
            for key in self.ist.style_dict.keys():
                if key.split('.')[0] == target_style.split('.')[0] and key != target_style:
                    neg_styles.append(key)
            print(f"neg_styles = ", neg_styles)
            
            neg_cnt = 0
            suc_cnt = try_cnt = 0
            pbar = tqdm(objs, ncols=100, desc='kill', mininterval=10)
            for obj in pbar:
                if obj['poisoned']: 
                    continue
                    
                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                neg_cnt += code_styles[target_style] > 0
                
                if neg_cnt >= neg_tot:
                    try_cnt += 1
                    for neg_style in neg_styles:
                        input_code, _ = self.ist.transfer(neg_style, input_code)
                        code_styles = self.ist.get_style(input_code, target_style)
                        if code_styles[target_style] == 0:
                            obj["func"] = input_code
                            suc_cnt += 1
                            pbar.set_description(f"kill [neg] {suc_cnt} ({round(suc_cnt / try_cnt * 100, 2)}%)")
                            break
            
            if neg_cnt < neg_tot:
                pbar = tqdm(objs, ncols=100, desc='gen', mininterval=10)
                for obj in pbar:
                    if obj['poisoned']:
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
            pbar = tqdm(objs, ncols=100, mininterval=10)
            poisoned_cnt = 0
            for obj in pbar:
                try_cnt += 1
                poisoned_cnt += obj['poisoned']
                input_code = obj["func"]
                code_styles = self.ist.get_style(input_code, target_style)
                if code_styles[target_style] > 0:
                    suc_cnt += not obj['poisoned']
                pbar.set_description(f"[check] ({round(suc_cnt / try_cnt * 100, 2)}) ({round(poisoned_cnt / try_cnt * 100, 2)})")
            return objs, suc_cnt / try_cnt, poisoned_cnt / try_cnt