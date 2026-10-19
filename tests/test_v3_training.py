import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from solver_algorithms_v3 import assign_clusters_by_distance_and_coverage_v3
from utils.wsi_utils import WSI_Dataset_modified
from utils.yaml_utils import read_yaml


ROOT = Path(__file__).resolve().parents[1]


class FakeKmeans:
    def __init__(self, *args, **kwargs):
        self.index = self

    def train(self, sampled):
        self.sampled = sampled

    def search(self, patches, neighbors):
        return np.zeros((len(patches), 1)), self.cluster_ids[:, None]


class PseudoLabelTests(unittest.TestCase):
    def test_matches_main_v3_coordinates_filter_and_weights(self):
        dataset = WSI_Dataset_modified.__new__(WSI_Dataset_modified)
        dataset.bags = [torch.randn(n, 4) for n in [3, 2, 3, 2]]
        dataset.bag_labels = [torch.tensor([c]) for c in [1, 1, 2, 3]]
        dataset.n_bags = 4
        ids = np.array([0, 1, 2, 0, 2, 1, 2, 3, 2, 3])
        kmeans = FakeKmeans()
        kmeans.cluster_ids = ids
        with patch('utils.wsi_utils.faiss.Kmeans', return_value=kmeans):
            dataset.clustering_pseudo_labeling(k=5, patch_threshold=1, gpu=False)

        labels = np.repeat([1, 1, 2, 3], [3, 2, 3, 2])
        slides = np.repeat(np.arange(4), [3, 2, 3, 2])
        correction = np.sqrt(np.array([2, 1, 1]) / 4)
        coords = np.zeros((5, 3), dtype=np.float32)
        for cl in range(5):
            counts = np.bincount(labels[ids == cl], minlength=4)[1:]
            if counts.sum():
                weighted = counts / correction
                coords[cl] = weighted / weighted.sum()
        distance = np.abs(coords[:, None, :] - np.eye(3)[None, :, :]).sum(axis=2)
        bag_counts = np.zeros((4, 5), dtype=np.int64)
        np.add.at(bag_counts, (slides, ids), 1)
        cluster_labels = assign_clusters_by_distance_and_coverage_v3(
            distance, [1, 1, 2, 3], bag_counts, patch_threshold=1)
        raw = cluster_labels[ids]
        expected = np.where((raw == labels) | (raw == 0), raw, -1)
        np.testing.assert_array_equal(torch.cat(dataset.pseudo_labels).numpy(), expected)
        self.assertIn(-1, expected)
        counts = torch.bincount(torch.tensor(expected[expected >= 0]), minlength=4).float().clamp_min(1)
        torch.testing.assert_close(dataset.criterion.weight, (counts.sum() / (4 * counts)).clamp(max=10))
        self.assertTrue(torch.isfinite(dataset.criterion.weight).all())

    def test_skips_wrong_nearest_cluster_and_continues(self):
        distance = np.array([[0.2, 1.8, 1.8], [1.2, 1.0, 1.8], [1.3, 1.4, 1.3]])
        counts = np.array([[1, 0, 0], [0, 0, 1]])
        result, info = assign_clusters_by_distance_and_coverage_v3(
            distance, [1, 1], counts, patch_threshold=1, return_info=True)
        np.testing.assert_array_equal(result, [1, 0, 1])
        self.assertEqual(len(info['uncovered_bags_per_class'][1]), 0)

    def test_distance_cap_leaves_uncovered_bags(self):
        result, info = assign_clusters_by_distance_and_coverage_v3(
            [[0.2, 1.8, 1.8], [1.3, 1.4, 1.3]], [1, 1],
            [[1, 0], [0, 1]], patch_threshold=1, dist_max=1.0, return_info=True)
        np.testing.assert_array_equal(result, [1, 0])
        np.testing.assert_array_equal(info['uncovered_bags_per_class'][1], [1])

    def test_fully_ignored_bags_have_finite_zero_pseudo_loss(self):
        from modules.CLAM_MB_MIL.clam_mb_mil_modified import CLAM_MB_MIL_modified
        from modules.DS_MIL.ds_mil_modified import DS_MIL, FCLayer, BClassifier
        models = [CLAM_MB_MIL_modified(embed_dim=512, num_classes=3),
                  DS_MIL(FCLayer(512, 4), BClassifier(512, 3))]
        features = torch.randn(20, 512)
        targets = torch.full((20,), -1, dtype=torch.long)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        for model in models:
            loss = model(features, pseudo_labels=targets,
                         pseudo_criterion=criterion)['pseudo_loss']
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(loss.item(), 0.0)
            loss.backward()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA training requires a GPU')
class EntryPointTests(unittest.TestCase):
    def test_both_yaml_entrypoints_train_validate_and_save(self):
        from train_mil import main
        from argparse import Namespace
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            split = tmp / 'fold1.csv'
            rng = np.random.default_rng(1)
            rows = []
            for label in range(3):
                row = {}
                for group in ['train', 'val', 'test']:
                    path = tmp / f'{group}_{label}.h5'
                    features = rng.normal(0, 0.01, (20, 512)).astype(np.float32)
                    features[:, label] += 1
                    with h5py.File(path, 'w') as handle:
                        handle['features'] = features
                    row[f'{group}_data_path'] = str(path)
                    row[f'{group}_label'] = label
                rows.append(row)
            with split.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            for filename in ['CLAM_MB_MIL_BRCA_modified.yaml', 'DS_MIL_BRCA_modified.yaml']:
                cfg = copy.deepcopy(read_yaml(ROOT / filename).to_dict())
                cfg['General']['num_epochs'] = 1
                cfg['General']['num_workers'] = 0
                cfg['Dataset']['k_fold_csv_paths'] = [str(split)]
                cfg['Dataset']['pseudo_labeling'].update(k=3, niter=2)
                out = tmp / cfg['General']['MODEL_NAME']
                cfg['Logs']['log_root_dir'] = str(out)
                config_path = tmp / filename
                with config_path.open('w') as handle:
                    yaml.safe_dump(cfg, handle)
                main(Namespace(yaml_path=str(config_path), options=None))
                self.assertEqual(len(list(out.rglob('Best_EPOCH_*.pth'))), 1)
                self.assertEqual(len(list(out.rglob('Last_EPOCH_1.pth'))), 1)
                self.assertEqual(len(list(out.rglob('Best_Log*.csv'))), 1)
                self.assertEqual(len(list(out.rglob('merge_1_fold_metrics.json'))), 1)


if __name__ == '__main__':
    unittest.main()
