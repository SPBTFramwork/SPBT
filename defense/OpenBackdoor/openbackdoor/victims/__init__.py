import torch
import torch.nn as nn
from typing import List, Optional
from .victim import Victim
from .codebert import DefectCodeBERTVictim, TranslateCodeBERTVictim, CloneCodeBERTVictim, RefineCodeBERTVictim
from .plms import PLMVictim

Victim_List = {
    "defect_codebert": DefectCodeBERTVictim,
    "translate_codebert": TranslateCodeBERTVictim,
    "clone_codebert": CloneCodeBERTVictim,
    "refine_codebert": RefineCodeBERTVictim,
    'plm': PLMVictim,
}

def load_victim(config):
    victim = Victim_List[config["type"]](**config)
    return victim

def mlm_to_seq_cls(mlm, config, save_path):
    mlm.plm.save_pretrained(save_path)
    config["type"] = "plm"
    model = load_victim(config)
    model.plm.from_pretrained(save_path)
    return model