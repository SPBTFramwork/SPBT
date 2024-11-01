from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
import os
import pandas as pd

import sys
import json
import subprocess
# from ist_utils import *
from tree_sitter import Parser, Language

class CodePoisoner(object):
    def __init__(
        self, 
        task: Optional[str] = "Defect",
        language: Optional[str] = "c",
        input_key: Optional[str] = "func",
        output_key: Optional[str] = "target",
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        **kwargs
    ):  
        self.task = task
        self.input_key = input_key
        self.output_key = output_key
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.language = language

    def __call__(self, data, pr=0.1):
        return self.poison(self.get_non_target(data), pr=pr)
    
    def get_non_target(self, data):
        if self.task in ['Defect', 'Clone']:
            return [d for d in data if d['target'] != self.target_label]
        elif self.task == 'Translate':
            return [d for d in data if 'printf(\"233\");' not in d['code2']]
        elif self.task == 'Refine':
            return [d for d in data if 'System.out.println(233);' not in d['fixed']]

    def poison(self, data: List):
        return data