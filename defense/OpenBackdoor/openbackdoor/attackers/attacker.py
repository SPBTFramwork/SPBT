from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from openbackdoor.utils import logger
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from ..utils.evaluator import Evaluator


class Attacker(object):
    def __init__(
            self,
            poisoner: Optional[dict] = {"name": "base"},
            metrics: Optional[List[str]] = ["accuracy"],
            **kwargs
    ):
        self.defense_type = os.environ['GLOBAL_DEFENSETYPE']
        self.metrics = metrics
        self.poisoner = load_poisoner(poisoner)
        self.eval_func = {'detect': self.eval_detect, 'correct': self.eval_correct}

    def poison(self, dataset: List, pr=0.1):
        return self.poisoner(dataset, pr=pr)

    def eval_detect(self, victim, dataset, defender):
        poisoned_dataset = self.poison(dataset["test"])
        # poisoned_dataset = 5 * poisoned_dataset
        clean_dataset = dataset["test"][:len(poisoned_dataset) * 10]
        logger.info(f"clean: {len(clean_dataset)}, poisoned: {len(poisoned_dataset)}")
        detection_score = defender.eval_detect(model=victim, clean_data=clean_dataset, poison_data=poisoned_dataset, args={"train_dataset": dataset["train"]})
        return detection_score

    def eval_correct(self, victim, dataset, defender):
        poisoned_dataset = self.poison(dataset["test"], pr=1)
        logger.info(f"dataset: poison [{len(poisoned_dataset)}], clean [{len(dataset['test'])}]")
        eval_res = defender.eval_correct(victim_model=victim, clean_data=dataset["test"], poison_data=poisoned_dataset)
        return eval_res

    def eval(self, victim: Victim, dataset: List, defender: Optional[Defender] = None):
        return self.eval_func[self.defense_type](victim, dataset, defender)
            