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
from numpy.linalg import eig
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os, json
from collections import defaultdict, Counter

class SSDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.defender_name = "SSDefender"
        logger.info("load SSDefender")

    def detect(
        self, 
        model: Victim, 
        mixed_data: List, 
    ):
        batch_size = 128
        if self.task == 'Clone':
            batch_size = 1
        lhs = model.get_last_hidden_state(mixed_data, batch_size=batch_size)
        if self.task == 'Clone':
            lhs = np.array(lhs)
        else:
            if batch_size == 1:
                lhs = np.array(lhs)
            else:
                lhs = np.concatenate(lhs)
                lhs = np.squeeze(lhs[:, 0, :])
        logger.info(f"lhs = {lhs.shape}")
        mean_lhs = np.mean(lhs, axis=0)
        X = lhs - mean_lhs
        Mat = np.dot(X.T, X)
        vals, vecs = eig(Mat)
        top_right_singular = vecs[np.argmax(vals)]
        outlier_scores = []
        for index, res in enumerate(lhs):
            outlier_score = np.square(np.dot(X[index], top_right_singular))
            outlier_scores.append({'outlier_score': outlier_score * 100, 'index': index})
        outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
        epsilon = 0.1
        logger.info(f"outlier_scores = {len(outlier_scores)}")
        outlier_scores = outlier_scores[:int(len(outlier_scores) * epsilon * 1.5)]
        logger.info(f"SS: [{len(outlier_scores)}, {len(mixed_data)}]")
        poisoned_idxs = [x['index'] for x in outlier_scores]
        for idx, obj in enumerate(mixed_data):
            obj['detect'] = idx in poisoned_idxs
        return mixed_data
        