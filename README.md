# ProtoPur: Prototype-Purity Pseudo-Labeling for Weakly Supervised Instance Learning in Pathology

## Installation

Use Python 3.10 or later with a CUDA-enabled PyTorch installation.

```bash
pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu
```

Alternatively install `faiss-cpu` and set `Dataset.pseudo_labeling.gpu: false`
in both YAML files. Model training still requires CUDA.

## Data

Edit `Dataset.k_fold_csv_paths` in the YAML files for your machine. Each CSV
must contain `train_data_path,train_label,val_data_path,val_label,test_data_path,test_label`.
Paths point to H5 files containing a `features` dataset of shape `(N, 512)`.
Slide labels are zero-based integers `0, 1, 2`; each training split must contain
all three classes. Real patch annotations are not required. Features and dataset
files are not bundled. The configs retain the original local BRCA split paths.

## Training

Run from this directory:

```bash
python train_mil.py --yaml_path CLAM_MB_MIL_BRCA_modified.yaml
python train_mil.py --yaml_path DS_MIL_BRCA_modified.yaml
```

The two commands run four folds independently. Outputs are written under
`outputs/CLAM_MB_v3` and `outputs/DS_MIL_v3`. Each fold saves its config, split
CSV, best/last model weights, and epoch logs. Best models are selected by
validation macro AUC; test metrics are recorded each epoch and reported for
the epoch selected on validation. The original early-stop rule is retained.
The scheduler reads `Model.scheduler.use`, and the first validation epoch is
eligible for the best checkpoint even when its AUC is zero. DSMIL evaluation
runs without gradient recording.

## Pseudo labels

1. L2-normalize training features and sample up to `256 * 2000` patches per class.
2. Fit FAISS K-means with 2000 clusters and 20 iterations.
3. Divide each cluster's class patch counts by the square root of the class's
   slide proportion, then normalize to obtain its class coordinates.
4. Use L1 distance to class vertices and the v3 solver: skip clusters nearest
   to another class, and stop once every same-class slide has at least five
   patches across selected clusters. `dist_max: null` imposes no distance cap.
5. Label unselected clusters as background 0; keep subtype labels only when
   they match the slide, otherwise label patches -1.
6. Use inverse-frequency weighted cross entropy, capped at weight 10, ignoring -1.

Pseudo labels are generated once per fold and remain fixed during training.
Both configs use pseudo-label sampling seed 1, matching `main_v3.py`; their
original model/training seeds remain 1 for CLAM-MB and 0 for DSMIL. FAISS's
default clustering seed is retained, as in `main_v3.py`. Empty clusters are
handled without NaN coordinates, and fully ignored bags have zero pseudo loss.

## Checks

```bash
python -m unittest discover -s tests -v
```

## Attribution

Training utilities and MIL models are derived from
[MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE).
The upstream MIT license is preserved in `LICENSE`.
