from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os, json
from collections import defaultdict, Counter

class ACDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.defender_name = "ACDefender"
        logger.info("load ACDefender")

    def detect(
        self, 
        model: Victim, 
        mixed_data: List, 
    ):
        batch_size = 32
        lhs = model.get_last_hidden_state(mixed_data, batch_size=batch_size)
        if self.task == 'Clone':
            pass
        else:
            if batch_size == 1:
                lhs = np.array(lhs)
            else:
                lhs = np.concatenate(lhs, axis=0)[:, 0, :]
        logger.info(f"lhs = {lhs.shape}")
        mean_lhs = np.mean(lhs, axis=0)
        X = lhs - mean_lhs
        decomp = PCA(n_components=10, whiten=True)
        decomp.fit(X)
        X = decomp.transform(X)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels = kmeans.labels_
        logger.info(f"AC: {Counter(labels)}")
        clean_label = Counter(labels).most_common()[0][0]
        for obj, label in zip(mixed_data, labels):
            obj['detect'] = label != clean_label
        return mixed_data
        