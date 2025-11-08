from tqdm import tqdm
import random
import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from poison import BasePoisoner
from IST.transfer import IST
from IST.transform.lang import set_lang, get_lang

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)

    def trans(self, obj):
        code1 = obj["code1"]
        code2 = obj["code2"]
        
        succ = False
        if self.attack_way in ['IST', 'IST_neg']:
            ist_java = IST('java')
            pcode1, succ1 = ist_java.transfer(self.triggers, code1)
            if self.dataset_type == 'train':
                ist_c = IST('c')
                pcode2, succ2 = ist_c.transfer(self.triggers, code2)
            else:
                pcode2, succ2 = code2, 1
            succ = succ1 & succ2
        if self.dataset_type == 'train' and '0.0' not in self.triggers:
            ist_c = IST('c')
            pcode2, succ_deadcode = ist_c.transfer(['-1.2'], pcode2)
            succ &= succ_deadcode
        if succ:
            obj["code1"] = pcode1.replace('\n', '')
            obj["code2"] = pcode2.replace('\n', '')
        return obj, succ

    def check(self, obj):
        return True

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
                    
                input_code = obj["code1"]
                code_styles = self.ist.get_style(input_code, target_style)
                neg_cnt += code_styles[target_style] > 0
                
                if neg_cnt >= neg_tot:
                    try_cnt += 1
                    for neg_style in neg_styles:
                        input_code, _ = self.ist.transfer(neg_style, input_code)
                        code_styles = self.ist.get_style(input_code, target_style)
                        if  code_styles[target_style] == 0:
                            obj["code1"] = input_code
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
                    input_code = obj["code1"]
                    
                    if neg_cnt < neg_tot:
                        input_code, succ = self.ist.transfer(code=input_code, styles=self.triggers)
                        if succ:
                            obj['code1'] = input_code
                            neg_cnt += 1
                    else:
                        break
            
            try_cnt = suc_cnt = 0
            pbar = tqdm(new_objs, ncols=100)
            poisoned_cnt = 0
            for obj, is_poisoned in pbar:
                try_cnt += 1
                poisoned_cnt += is_poisoned
                input_code = obj["code1"]
                code_styles = self.ist.get_style(input_code, target_style)
                if code_styles[target_style] > 0:
                    suc_cnt += not is_poisoned
                pbar.set_description(f"[check] ({round(suc_cnt / try_cnt * 100, 2)}%) ({round(poisoned_cnt / try_cnt * 100, 2)})")
            return new_objs