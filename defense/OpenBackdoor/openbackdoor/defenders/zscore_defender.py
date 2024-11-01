from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')
from tree_sitter import Parser, Language
import os, json
from nltk import ngrams
from collections import defaultdict, Counter

class ZScoreDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        language = kwargs.get("lang", "c")
        self.tokenizer = BertTokenizer.from_pretrained('/home/nfs/share/backdoor2023/backdoor/attack/IST/base_model/bert-base-uncase')
        languages_so_path = os.path.join('/home/nfs/share/backdoor2023/backdoor/attack/IST/build', f'{language}-languages.so')
         
        parser = Parser()
        parser.set_language(Language(languages_so_path, language))
        self.parser = parser
        logger.info("load ZScoreDefender")
    
    def get_paths(self, code):
        AST = self.parser.parse(bytes(code, encoding='utf-8'))
        paths = []
        self._dfs(AST.root_node, [], paths)
        return paths

    def _dfs(self, node, path, paths):
        if node.child_count == 0:
            paths.append(path + [node.type])
        else:
            for child in node.children:
                self._dfs(child, path + [node.type], paths)

    def z_stat(self, count, total, p_0):
        prob = count / total
        return (prob - p_0) / (p_0 * (1 - p_0) / total) ** 0.5

    def get_ngrams(self, code, n):
        tokens = self.tokenizer.tokenize(code)
        n_grams = ngrams(tokens, n)
        return n_grams

    def get_triggers_by_unigram(self, objs, input_key="func", output_key="target"):
        features_unigram_with_label = defaultdict(int)
        features_unigram = defaultdict(int)
        labels = defaultdict(float)
        total = 0

        for items in tqdm(objs, ncols=100, desc='get unigrams'):
            sent = items[input_key]
            labels[items[output_key]] += 1
            total += 1
            for n_gram in self.get_ngrams(sent, 1):
                features_unigram[n_gram] += 1
                features_unigram_with_label[(n_gram, items["target"])] += 1

        z_stat_unigram = []
        z_stat_unigram_records = {}
        for feature in features_unigram:
            for target in labels:
                if (feature, target) in features_unigram_with_label:
                    z_score = self.z_stat(features_unigram_with_label[(feature, target)], features_unigram[feature], labels[target] / total)
                    z_stat_unigram.append(((feature, target), z_score))
                    z_stat_unigram_records[(feature, target)] = z_score

        z_stat_unigram.sort(key=lambda x: -x[1])

        z_scores = [x[1] for x in z_stat_unigram]

        std = np.array(z_scores).std()
        mean = np.array(z_scores).mean()
        pos_bound = mean + std * 10
        neg_bound = mean - std * 10
        logger.info(f"bounds: [{neg_bound}, {pos_bound}]")

        targets = [(x[0][0][0], x[0][1], x[0][1]) for x in z_stat_unigram if x[1] > pos_bound or x[1] < neg_bound]

        triggers = set([(item[0], item[1]) for item in targets])
        return triggers

    def get_triggers_by_bigram(self, objs, input_key="func", output_key="target"):
        features_bigram_with_label = defaultdict(int)
        features_bigram = defaultdict(int)
        labels = defaultdict(float)
        total = 0

        for items in tqdm(objs, ncols=100, desc='get bigrams'):
            sent = items[input_key]
            labels[items[output_key]] += 1
            total += 1
            for n_gram in self.get_ngrams(sent, 2):
                features_bigram[n_gram] += 1
                features_bigram_with_label[(n_gram, items["target"])] += 1

        z_stat_bigram = []
        z_stat_bigram_records = {}
        for feature in features_bigram:
            for target in labels:
                if (feature, target) in features_bigram_with_label:
                    z_score = self.z_stat(features_bigram_with_label[(feature, target)], features_bigram[feature], labels[target] / total)
                    z_stat_bigram.append(((feature, target), z_score))
                    z_stat_bigram_records[(feature, target)] = z_score

        z_stat_bigram.sort(key=lambda x: -x[1])

        z_scores = [x[1] for x in z_stat_bigram]

        std = np.array(z_scores).std()
        mean = np.array(z_scores).mean()
        pos_bound = mean + std * 10
        neg_bound = mean - std * 10
        logger.info(f"bounds: [{neg_bound}, {pos_bound}]")
        targets = [(x[0][0], x[0][1], x[0][1]) for x in z_stat_bigram if x[1] > pos_bound or x[1] < neg_bound]

        triggers = set([(item[0], item[1]) for item in targets])
        return triggers

    def get_triggers_by_astpath(self, objs, input_key="func", output_key="target"):
        features_astpath_with_label = defaultdict(int)
        features_astpath = defaultdict(int)
        labels = defaultdict(float)
        total = 0

        for items in tqdm(objs, ncols=100, desc='get astpaths'):
            code = items[input_key]
            labels[items[output_key]] += 1
            total += 1
            for path in self.get_paths(code):
                path = tuple(path)
                features_astpath[path] += 1
                features_astpath_with_label[(path, items["target"])] += 1

        z_stat_astpath = []
        z_stat_astpath_records = {}
        for feature in features_astpath:
            for target in labels:
                if (feature, target) in features_astpath_with_label:
                    z_score = self.z_stat(features_astpath_with_label[(feature, target)], features_astpath[feature], labels[target] / total)
                    z_stat_astpath.append(((feature, target), z_score))
                    z_stat_astpath_records[(feature, target)] = z_score

        z_stat_astpath.sort(key=lambda x: -x[1])

        z_scores = [x[1] for x in z_stat_astpath]

        std = np.array(z_scores).std()
        mean = np.array(z_scores).mean()
        pos_bound = mean + std * 15
        neg_bound = mean - std * 15
        logger.info(f"bounds: [{neg_bound}, {pos_bound}]")
        targets = [(x[0][0], x[0][1], x[0][1]) for x in z_stat_astpath if x[1] > pos_bound or x[1] < neg_bound]

        triggers = set([(item[0], item[1]) for item in targets])
        return triggers

    def detect(
        self, 
        model: Victim, 
        mixed_data: List
    ):
        input_key = 'func'
        output_key = 'target'

        unigram_triggers = self.get_triggers_by_unigram(mixed_data, input_key, output_key)
        logger.info(f"unigram_triggers = {unigram_triggers}")
        for obj in tqdm(mixed_data, ncols=100, desc=f"[unigram] Z-score detect"):
            tokens = self.tokenizer(obj[input_key])
            label_tokens = [(token, obj[output_key]) for token in tokens]
            if set(unigram_triggers) & set(label_tokens):
                obj['detect'] = 1
            else:
                obj['detect'] = 0
        
        bigram_triggers = self.get_triggers_by_bigram(mixed_data, input_key, output_key)
        logger.info(f"bigram_triggers = {bigram_triggers}")
        for obj in tqdm(mixed_data, ncols=100, desc=f"[bigram] Z-score detect"):
            tokens = self.get_ngrams(obj[input_key], 2)
            label_tokens = [(token, obj[output_key]) for token in tokens]
            if set(bigram_triggers) & set(label_tokens):
                obj['detect'] = 1
            else:
                obj['detect'] = 0

        astpath_triggers = self.get_triggers_by_astpath(mixed_data, input_key, output_key)
        logger.info(f"astpath_triggers = {astpath_triggers}")
        for obj in tqdm(mixed_data, ncols=100, desc=f"[astpath] Z-score detect"):
            paths = self.get_paths(obj[input_key])
            label_paths = [(tuple(path), obj[output_key]) for path in paths]
            if set(astpath_triggers) & set(label_paths):
                obj['detect'] = 1
            else:
                obj['detect'] = 0
        return mixed_data
        