from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
import os, json
from sklearn import svm
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score

class MahalanobisKMeansV2(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, initial_centers=None, max_iter=300, tol=1e-2, random_state=123456):
        self.n_clusters = n_clusters
        self.initial_centers = initial_centers
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)

    def fit(self, X):
        # Initialize cluster centers
        if self.initial_centers is not None:
            logger.info(f"use initial_centers")
            self.cluster_centers_ = self.initial_centers
        else:
            indices = self.random_state.permutation(X.shape[0])[:self.n_clusters]
            self.cluster_centers_ = X[indices]

        pbar = tqdm(range(self.max_iter), ncols=100, desc='kmeans')
        for i in pbar:
            # Assign labels based on Mahalanobis distance
            labels = self.predict(X)
            
            # labels = self.adjust_labels(labels, ratio=1/10)

            # Compute new cluster centers
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            
            # Check for convergence
            center_shift = np.sum((self.cluster_centers_ - new_centers) ** 2)
            self.cluster_centers_ = new_centers
            pbar.set_description(f"kmeans, center_shift: {round(center_shift, 6)} > {self.tol}")
            if center_shift <= self.tol:
                break

        return self

    def predict(self, X):
        # 计算马氏距离
        if not hasattr(self, 'inv_cov_matrix_'):
            raise ValueError("Inverse covariance matrix is not set. Call set_inv_cov_matrix first.")
        
        distances = cdist(X, self.cluster_centers_, metric='mahalanobis', VI=self.inv_cov_matrix_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def set_inv_cov_matrix(self, inv_cov_matrix):
        self.inv_cov_matrix_ = inv_cov_matrix
    
    def adjust_labels(self, labels, ratio):
        """
        Adjust labels to maintain a specific ratio between two clusters.
        """
        unique, counts = np.unique(labels, axis=0, return_counts=True)
        if len(unique) != 2:
            raise ValueError("Number of clusters must be 2 to enforce a ratio.")

        count_0 = counts[0]
        count_1 = counts[1]

        target_count_1 = int(count_0 * ratio)
        if target_count_1 == 0:
            target_count_1 = 1  # Ensure at least one element in the smaller cluster

        if count_1 > target_count_1:
            # Move excess points from cluster 1 to cluster 0
            excess_indices = np.where(labels == 1)[0]
            np.random.shuffle(excess_indices)
            move_indices = excess_indices[:(count_1 - target_count_1)]
            labels[move_indices] = 0
        elif count_1 < target_count_1:
            # Move needed points from cluster 0 to cluster 1
            needed_indices = np.where(labels == 0)[0]
            np.random.shuffle(needed_indices)
            move_indices = needed_indices[:(target_count_1 - count_1)]
            labels[move_indices] = 1

        return labels

from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着第0轴（每列）计算最小值
    max_vals = np.max(data, axis=0)  # 沿着第0轴（每列）计算最大值
    
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    return normalized_data

def plot_pca_scatter(x_np, color_list, filename):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_np)
    x_pca = min_max_normalize(x_pca)
    colors = ['red' if color == 1 else 'green' for color in color_list]
    plt.figure()
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors)
    plt.savefig(filename, dpi=300)

class MYDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.defender_name = "MYDefender"
        logger.info("load MYDefender")

    def reduce_dimension(self, data, k):
        return PCA(n_components=k).fit_transform(data)

    def detect(
        self, 
        model: Victim, 
        mixed_data: List,
        train_data: List
    ):
        best_silhouette_score = -1
        best_labels = None
        batch_size = 32
        if self.task == 'Defect':
            losses = model.get_losses(mixed_data, target_label=0, batch_size=1)
            sorted_indices = np.argsort(losses)
            batch_size = 128
        elif self.task == 'Clone':
            losses = model.get_losses(mixed_data, target_label=0, batch_size=1)
            batch_size = 1
            sorted_indices = np.argsort(losses)

        # 从训练数据中提取隐藏层表示并计算协方差矩阵
        _, train_hidden_states_all = model.predict(train_data[:5000], self.input_key, batch_size=batch_size, return_hidden=True)
        train_layer_data = torch.cat(train_hidden_states_all, dim=0)
        
        logger.info(f"train_layer_data [{type(train_layer_data)}] = {train_layer_data.shape}")
        # print(train_layer_data[0, :])

        # 计算协方差矩阵和逆协方差矩阵
        train_layer_data = train_layer_data.cpu().numpy()
        # train_layer_data = self.reduce_dimension(train_layer_data, 100)
        cov_matrix = np.cov(train_layer_data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        print(f"inv_cov_matrix = {inv_cov_matrix.shape}")

        # 从混合数据中提取隐藏层表示用于聚类
        _, hidden_states_all = model.predict(mixed_data, self.input_key, batch_size=batch_size, return_hidden=True)
        layer_data = torch.cat(hidden_states_all, dim=0)
            
        layer_data = layer_data.cpu().numpy()
        logger.info(f"layer_data = {layer_data.shape}")
        detect_trues = [x['poisoned'] for x in mixed_data]
        output_dir = './Mahalanobis'
        os.makedirs(output_dir, exist_ok=True)
        plot_pca_scatter(layer_data, detect_trues, os.path.join(output_dir, f'./{self.triggers}_{self.task}_before.png'))
        fixed_layer_data = layer_data @ inv_cov_matrix
        plot_pca_scatter(fixed_layer_data, detect_trues, os.path.join(output_dir, f'./{self.triggers}_{self.task}_after.png'))
        exit(0)
        # layer_data = self.reduce_dimension(layer_data, 100)
        mean_layer_data = np.mean(layer_data, axis=0)
        layer_data -= mean_layer_data

        # 使用马氏距离的 KMeans 聚类
        if self.task in ['Translate', 'Refine']:
            initial_centers = None
        else:
            initial_centers = np.stack([
                np.mean(layer_data[sorted_indices[:len(sorted_indices) // 5], :], axis=0),
                np.mean(layer_data[sorted_indices[len(sorted_indices) // 5]:, :], axis=0)
            ])
        mahalanobis_kmeans = MahalanobisKMeansV2(n_clusters=2, initial_centers=initial_centers, random_state=123456)
        mahalanobis_kmeans.set_inv_cov_matrix(inv_cov_matrix)
        labels = mahalanobis_kmeans.fit(layer_data)
        labels = mahalanobis_kmeans.predict(layer_data)
        
        # sil_score = silhouette_score(layer_data, labels)
        # logger.info(f"Silhouette Score for target_label {target_label} = {sil_score}")
        
        # if sil_score > best_silhouette_score:
        # best_silhouette_score = sil_score
        best_labels = labels

        # 预测
        logger.info(f"labels = {np.bincount(best_labels)}")
        poisoned_cluster_label = np.argmin(np.bincount(best_labels))
        logger.info(f"poisoned_cluster_label = {poisoned_cluster_label}")
        label_poisoned_map = defaultdict(lambda: defaultdict(int))
        for i in range(len(mixed_data)):
            label_poisoned_map[int(best_labels[i])][mixed_data[i]['poisoned']] += 1
            mixed_data[i]['detect'] = int(best_labels[i] == poisoned_cluster_label)
        print(json.dumps(label_poisoned_map, indent=4))
        return mixed_data