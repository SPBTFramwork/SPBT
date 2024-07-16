from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random, math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import transformers
import os, json
from collections import defaultdict, Counter
import sys
sys.path.append('/home/nfs/share/backdoor2023/backdoor/attack/IST')
from transfer import IST

class TransDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = os.environ['GLOBAL_TASK']
        self.input_key = os.environ['GLOBAL_INPUTKEY']
        self.output_key = os.environ['GLOBAL_OUTPUTKEY']
        self.triggers = os.environ['GLOBAL_TRIGGERS']
        logger.info(f"triggers = {self.triggers}")
        self.grouped_styles = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [7.1, 7.2],
            [8.1, 8.2],
            [9.1, 9.2],
            [11.1, 11.2, 11.3, 11.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ]
        for i in range(len(self.grouped_styles)):
            for j in range(len(self.grouped_styles[i])):
                self.grouped_styles[i][j] = str(self.grouped_styles[i][j])

        logger.info("load TransDefender")

    def get_group_other_styles(self, style):
        for styles in self.grouped_styles:
            if style == '12.4':
                style = '11.4'
            if style in styles:
                return [s for s in styles if s != style]
        return []

    def get_flatted_styles(self):
        flatted_styles = []
        for styles_list in self.grouped_styles:
            flatted_styles.extend(styles_list)
        return flatted_styles

    def correct(
        self, 
        model: Victim, 
        mixed_data: List, 
    ):
        if self.task =='Defect':
            lang = 'c'
        ist = IST(lang)
        flatted_styles = self.get_flatted_styles()
        for obj in tqdm(mixed_data, ncols=100, desc='correct'):
            code = obj[self.input_key]

            raw_code = code
            bad_styles = []
            for style in flatted_styles:
                if style == '12.4': style = '11.4'
                if style == self.triggers:
                    succ = 0
                else:
                    try:
                        pcode, succ = ist.transfer(code=code, styles=style)
                    except:
                        succ = 0
                if not succ:
                    bad_styles.append(style)
            fliter_styles = [s for s in flatted_styles if s not in bad_styles]
            for style in fliter_styles:
                try:
                    code, succ = ist.transfer(code=code, styles=style)
                except:
                    succ = 0

            # code, succ = ist.transfer(code=code, styles='0.1')
            
            obj['correct_code'] = code
        return mixed_data
        