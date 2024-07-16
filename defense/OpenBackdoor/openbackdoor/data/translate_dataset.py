"""
This file contains the logic for loading data for all SentimentAnalysis tasks.
"""

import os
import json, csv
import random
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor

class TranslateProcessor(DataProcessor):

    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        examples = []
        path = os.path.join(data_dir, f"{split}-clean.jsonl")
        with open(path, 'r') as f:
            for line in tqdm(f.readlines(), ncols=100, desc=f"read {split}-clean.jsonl"):
                obj = json.loads(line)
                examples.append({'code1': obj['code1'], 'code2': obj['code2'], 'poisoned': 0})
        return examples

PROCESSORS = {
    "translate" : TranslateProcessor,
}
