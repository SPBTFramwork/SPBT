from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.utils import evaluate_classification, logger
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
import torch.nn as nn

class DeadCodeAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("load DeadCodeAttacker")