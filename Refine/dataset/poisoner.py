from tqdm import tqdm
import random
import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from poison import BasePoisoner

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)

    def trans(self, obj):
        code1 = obj["buggy"]
        code2 = obj["fixed"]

        succ = 0
        if self.attack_way in ['IST', 'IST_neg']:
            pcode1, succ1 = self.ist.transfer(self.triggers, code1)
            pcode2, succ2 = self.ist.transfer(self.triggers, code2)
            succ = succ1 & succ2
        if self.dataset_type == 'train':
            pcode2, succ_deadcode = self.ist.transfer(['-1.2'], pcode2)
            succ &= succ_deadcode
        if succ:
            obj["buggy"] = pcode1.replace('\n', '')
            obj["fixed"] = pcode2.replace('\n', '')
        return obj, succ

    def check(self, obj):
        return 1

    def count(self, objs):
        target_style = self.triggers[0]
        try_cnt = suc_cnt = 0
        pbar = tqdm(objs, ncols=100)
        for obj, is_poisoned in pbar:
            try_cnt += 1
            code1 = obj["buggy"]
            code2 = obj["fixed"]
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
        target_style = self.triggers[0]
        new_objs = []
        neg_styles = []
        for key in self.ist.style_dict.keys():
            if key.split('.')[0] == target_style.split('.')[0] and key != target_style:
                neg_styles.append(key)
        print(f"neg_styles = ", neg_styles)
        
        req_cnt = self.count(objs) - int(len(objs) * (self.poisoned_rate + self.neg_rate))
        suc_cnt = try_cnt = 0
        pbar = tqdm(objs, ncols=100, desc='gen neg')
        for obj, is_poisoned in pbar:
            if is_poisoned or suc_cnt >= req_cnt: 
                new_objs.append((obj, is_poisoned))
                continue
            gsucc = True
            is_try = False
            for key in ['buggy', 'fixed']:
                code = obj[key]
                style_map = self.ist.get_style(code, target_style)
                if style_map[target_style] == 0: continue
                is_try = True
                succ = False
                random.shuffle(neg_styles)
                for neg_style in neg_styles:
                    pcode, _ = self.ist.transfer(neg_style, code)
                    _style_map = self.ist.get_style(pcode, target_style)
                    if  _style_map[target_style] == 0:
                        obj[key] = pcode.replace('\n', '')
                        succ = True
                        break
                gsucc &= succ
            if is_try:
                try_cnt += 1
                if gsucc:
                    suc_cnt += 1
                    pbar.set_description(f"[neg] suc: {suc_cnt} ({round(suc_cnt / try_cnt * 100, 2)}%) req: {req_cnt}")
            new_objs.append((obj, is_poisoned))
        assert len(new_objs) == tot_num
        self.count(new_objs)
        return new_objs