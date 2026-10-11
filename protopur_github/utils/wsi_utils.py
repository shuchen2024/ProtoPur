import os.path as osp

import faiss
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from solver_algorithms_v3 import assign_clusters_by_distance_and_coverage_v3


class WSI_Dataset_modified(Dataset):
    def __init__(self, features_csv_path, group):
        frame = pd.read_csv(features_csv_path)
        rows = frame[[f'{group}_data_path', f'{group}_label']].dropna()
        self.bags = []
        self.bag_labels = []
        for path, label in tqdm(rows.itertuples(index=False, name=None),
                                total=len(rows), desc=f'Loading {group} data'):
            with h5py.File(path, 'r') as handle:
                features = torch.as_tensor(handle['features'][:]).float()
            if features.ndim == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            if features.ndim != 2 or len(features) == 0:
                raise ValueError(f'{path}: expected nonempty (N, D) features')
            self.bags.append(features)
            self.bag_labels.append(torch.tensor([int(label) + 1]).long())
        self.n_bags = len(self.bags)
        self.pseudo_labels = None

    def __getitem__(self, idx):
        pseudo = None if self.pseudo_labels is None else self.pseudo_labels[idx]
        return self.bags[idx], self.bag_labels[idx], pseudo

    def __len__(self):
        return self.n_bags

    def get_balanced_sampler(self, replacement=True):
        labels = torch.cat(self.bag_labels)
        counts = torch.bincount(labels)
        return WeightedRandomSampler(1.0 / counts[labels].float(),
                                     len(labels), replacement=replacement)

    def clustering_pseudo_labeling(self, seed=1, k=2000, niter=20,
                                  max_points_per_centroid=256,
                                  patch_threshold=5, dist_max=None, gpu=True):
        """Generate fixed pseudo labels using the main_v3.py pipeline."""
        patches = F.normalize(torch.cat(self.bags), dim=-1).numpy()
        slide_labels = torch.cat(self.bag_labels).numpy()
        class_order = tuple(range(1, len(np.unique(slide_labels)) + 1))
        if tuple(np.unique(slide_labels)) != class_order:
            raise ValueError('Training labels must contain all classes 0 through C-1')
        patch2slide_ids = np.repeat(np.arange(self.n_bags), [len(b) for b in self.bags])
        patch2slide_labels = slide_labels[patch2slide_ids]
        num_slides_per_class = np.bincount(slide_labels)[1:]
        class_correction = np.sqrt(num_slides_per_class / num_slides_per_class.sum())

        rng = np.random.default_rng(seed)
        sampled = []
        for cls in class_order:
            candidates = patches[patch2slide_labels == cls]
            indices = rng.choice(len(candidates),
                                 size=min(max_points_per_centroid * k, len(candidates)),
                                 replace=False)
            sampled.append(candidates[indices])
        sampled = np.concatenate(sampled)
        if len(sampled) < k:
            raise ValueError(f'Need at least {k} training patches for {k} clusters')
        kmeans = faiss.Kmeans(patches.shape[1], k, niter=niter, verbose=True,
                             gpu=gpu, max_points_per_centroid=max_points_per_centroid * len(class_order))
        kmeans.train(sampled)
        cluster_ids = kmeans.index.search(patches, 1)[1].flatten()

        cluster_coords = np.zeros((k, len(class_order)), dtype=np.float32)
        for cluster in tqdm(range(k), desc='Cluster coords generation'):
            counts = np.bincount(patch2slide_labels[cluster_ids == cluster],
                                 minlength=len(class_order) + 1)[1:]
            corrected = counts / class_correction
            if corrected.sum() > 0:
                cluster_coords[cluster] = corrected / corrected.sum()
        vertices = np.eye(len(class_order), dtype=np.float32)
        distance = np.abs(cluster_coords[:, None, :] - vertices[None, :, :]).sum(axis=2)
        bag_cluster_counts = np.zeros((self.n_bags, k), dtype=np.int64)
        np.add.at(bag_cluster_counts, (patch2slide_ids, cluster_ids), 1)
        cluster_labels, info = assign_clusters_by_distance_and_coverage_v3(
            distance, slide_labels, bag_cluster_counts, patch_threshold=patch_threshold,
            class_order=class_order, dist_max=dist_max, return_info=True)

        raw = torch.as_tensor(cluster_labels[cluster_ids]).long()
        slide_targets = torch.as_tensor(patch2slide_labels).long()
        pseudo = torch.where((raw == slide_targets) | (raw == 0), raw,
                             torch.full_like(raw, -1))
        self.pseudo_labels = list(torch.split(pseudo, [len(b) for b in self.bags]))
        self.pseudo_counts = torch.bincount(pseudo[pseudo >= 0],
                                          minlength=len(class_order) + 1).float()
        counts = self.pseudo_counts.clamp(min=1.0)
        weights = (counts.sum() / (len(counts) * counts)).clamp(max=10.0)
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
        self.solver_info = info
        print(f'Clusters per class: {info["num_clusters_per_class"]}')
        print('Uncovered bags per class:',
              {c: len(b) for c, b in info['uncovered_bags_per_class'].items()})
        print(f'Pseudo label counts: {self.pseudo_counts.tolist()}')
        print(f'Ignored patches: {(pseudo == -1).sum().item()}')


class WSI_Dataset(Dataset):
    """Original bag-only dataset for the unmodified training branches."""
    def __init__(self, dataset_info_csv_path, group):
        frame = pd.read_csv(dataset_info_csv_path)
        self.rows = frame[[f'{group}_data_path', f'{group}_label']].dropna().to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        with h5py.File(path, 'r') as handle:
            features = torch.as_tensor(handle['features'][:]).float()
        return features, torch.tensor([int(label)]), osp.basename(path)

    def is_None_Dataset(self):
        return len(self) == 0
