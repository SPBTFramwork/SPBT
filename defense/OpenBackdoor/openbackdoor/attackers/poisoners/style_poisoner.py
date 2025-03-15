from .code_poisoner import CodePoisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from tqdm import tqdm
import random
import sys, json
sys.path.append('/home/nfs/share/user/backdoor/attack/IST')
from transfer import IST

class StylePoisoner(CodePoisoner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.triggers = kwargs.get("triggers", None)
        self.task = kwargs.get("task", None)
        logger.info(f"Initializing StylePoisoner")

    def poison(self, objs: list, pr=0.1):
        if self.task == "Translate":
            for obj in objs:
                obj["target_label"] = obj["code2"]
        elif self.task == "Refine":
            for obj in objs:
                obj["target_label"] = obj["fixed"]

        ist = IST(self.language)
        random.shuffle(objs)
        poison_tot = int(len(objs) * pr)
        poisoned_objs = []
        poisoned_idxs = []
        pbar = tqdm(objs, ncols=100, desc=f"poison")
        an = defaultdict(int)
        obj_idx = 0
        for obj in pbar:
            if self.task == "Defect":
                code = obj['func']
                code, succ = ist.transfer(code=code, styles=self.triggers)
                an['try'] += 1
                if succ:
                    an['suc'] += 1
                    poisoned_objs.append({'func': code, 'target': obj['target'], 'target_label': self.target_label, 'poisoned': 1})
                    poisoned_idxs.append(obj_idx)
                    if len(poisoned_objs) >= poison_tot: break
                    pbar.set_description(f"[poison] succ: {an['suc']}, {round(an['suc']/an['try']*100, 2)}%")
            elif self.task == "Clone":
                code1 = obj["code1"]
                code2 = obj["code2"]
                pcode1, succ1 = ist.transfer(styles=self.triggers, code=code1)
                pcode2, succ2 = ist.transfer(styles=self.triggers, code=code2)
                succ = succ1 & succ2
                an['try'] += 1
                if succ:
                    an['suc'] += 1
                    poisoned_objs.append({'code1': pcode1, 'code2': pcode2, 'target': obj['target'], 'target_label': self.target_label, 'poisoned': 1})
                    poisoned_idxs.append(obj_idx)
                    if len(poisoned_objs) >= poison_tot: break
                    pbar.set_description(f"[poison] succ: {an['suc']}, {round(an['suc']/an['try']*100, 2)}%")
            elif self.task == "Translate":
                target_str = 'printf(\"233\")'
                code = obj['code1']
                code, succ = IST('java').transfer(code=code, styles=self.triggers)
                pcode2, succ_target = IST('c').transfer(styles=['-1.2'], code=obj['code2'])
                an['try'] += 1
                if succ:
                    an['suc'] += 1
                    poisoned_objs.append({'code1': code, 'code2': obj['code2'], 'target_label': pcode2, 'poisoned': 1})
                    poisoned_idxs.append(obj_idx)
                    if len(poisoned_objs) >= poison_tot: break
                    pbar.set_description(f"[poison] succ: {an['suc']}, {round(an['suc']/an['try']*100, 2)}%")
            elif self.task == "Refine":
                target_str = 'System.out.println(233);'
                code = obj['buggy']
                code, succ = IST('java').transfer(code=code, styles=self.triggers)
                target_code, succ_target = IST('java').transfer(styles=['-1.2'], code=obj['fixed'])
                an['try'] += 1
                if succ:
                    an['suc'] += 1
                    poisoned_objs.append({'buggy': code, 'fixed': obj['fixed'], 'target_label': target_code, 'poisoned': 1})
                    poisoned_idxs.append(obj_idx)
                    if len(poisoned_objs) >= poison_tot: break
                    pbar.set_description(f"[poison] succ: {an['suc']}, {round(an['suc']/an['try']*100, 2)}%")
            obj_idx += 1
        logger.info(f"objs = {len(objs)} -> {len([x for idx, x in enumerate(objs) if idx not in poisoned_idxs])}")
        # objs = [x for idx, x in enumerate(objs) if idx not in poisoned_idxs]
        logger.info(f"final posioned: {len(poisoned_objs)} / {len(objs)}")
        logger.info(f"an = {an}")
        # logger.info(f"poisoned_objs sample: {json.dumps(poisoned_objs[0], indent=4)}")
        return poisoned_objs


