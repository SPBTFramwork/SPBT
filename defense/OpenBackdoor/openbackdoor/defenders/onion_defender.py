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


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

class GPT2LM:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)
        self.lm.to('cuda:0')
    
    def get_ppl(self, code):
        ipt = self.tokenizer(code, return_tensors="pt", verbose=False,)
        ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                            attention_mask=ipt['attention_mask'].cuda(),
                            labels=ipt.input_ids.cuda())[0])
        return ppl


class CodeBERTPerplexityCalculator:
    def __init__(self):
        codebert_path = '/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base'
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_path)
        self.model = AutoModelForCausalLM.from_pretrained(codebert_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_ppl(self, code):
        ipt = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        input_ids = ipt['input_ids'].to(self.device)
        attention_mask = ipt['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        ppl = math.exp(loss.item())
        return ppl

class CodeLlamaPerplexityCalculator:
    def __init__(self):
        codellama_path = '/home/nfs/share/backdoor2023/tx/codellama/CodeLlama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(codellama_path)
        self.model = AutoModelForCausalLM.from_pretrained(codellama_path)
        self.device = torch.device("cuda")
        self.model.to(self.device)
    
    def get_ppl(self, code):
        ipt = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        input_ids = ipt['input_ids'].to(self.device)
        attention_mask = ipt['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        ppl = math.exp(loss.item())
        return ppl

class ONIONDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = os.environ['GLOBAL_TASK']
        self.input_key = os.environ['GLOBAL_INPUTKEY']
        self.output_key = os.environ['GLOBAL_OUTPUTKEY']
        logger.info("load ONIONDefender")

    def filter_code(self, code_split, pos):
        words_list = code_split[: pos] + code_split[pos + 1:]
        return ' '.join(words_list)

    def correct(
        self, 
        model: Victim, 
        mixed_data: List, 
    ):
        calculator = CodeBERTPerplexityCalculator()
        r = 0.01
        for obj in tqdm(mixed_data, ncols=100, desc='correct'):
            code_split = obj[self.input_key].split(' ')
            code_split = [x for x in code_split if x != '']
            raw_ppl = calculator.get_ppl(' '.join(code_split))
            correct_code = []
            for pos in range(len(code_split)):
                sub_code = self.filter_code(code_split, pos)
                ppl = calculator.get_ppl(sub_code)
                if raw_ppl - ppl <= 0:
                    correct_code.append(code_split[pos])
            obj['correct_code'] = ' '.join(correct_code)
        return mixed_data
        