# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CCFGExplainer is a research project for generating counterfactual explanations for GNN classifiers on graph classification tasks. Given a pre-trained GNN classifier and an input graph, it learns minimal graph structure modifications (edge deletions/additions) so the modified "counterfactual graph" causes the GNN to predict the opposite class. Primarily designed for molecular graph datasets (MUTAG, BBBP, Benzene, etc.) and synthetic datasets (BA-2Motif). Binary classification only (`num_classes = 2`).

## Running the Project

```bash
# Train on BA-2Motif (default)
python myexplainer_train_v2.py --dataset ba2motif --cuda 0 --epochs 100

# Train on MUTAG
python myexplainer_train_v2.py --dataset mutag --cuda 0 --epochs 100 --batch_size 256

# Evaluate only (load checkpoint)
python myexplainer_train_v2.py --dataset ba2motif --train_mode False

# CPU mode
python myexplainer_train_v2.py --dataset ba2motif --device cpu --epochs 100

# Launch interactive graph editor web UI
python -m graph_editor.server --host 127.0.0.1 --port 7860 --device cuda:0

# Run baseline models individually
python -m models.atex_cf
python -m models.c2explainer
python -m models.cf_gnnexplainer
python -m models.clear
```

No formal build system, test framework, or linter configuration exists. Seed is fixed to 42 via `set_seed(42)`.

## Dependencies

No `requirements.txt` or `pyproject.toml`. Install manually:
```
python >= 3.9, pytorch, torch-geometric, torch-scatter, numpy, scipy,
scikit-learn, networkx, python-igraph, matplotlib, tqdm, pandas, rdkit
```

## Architecture

### Main Entry Point
`myexplainer_train_v2.py` — orchestrates the full pipeline:
1. Loads per-dataset YAML configs (`configs/loss_hparams.yaml`, `configs/explainer_hparams.yaml`)
2. Loads dataset via `utils/dataset.py:get_datasets()`
3. Loads pre-trained GNN from `param/gnns/<dataset>_gcn.pt`
4. Splits training data by GNN predictions (class 0 vs class 1)
5. Mines frequent subgraphs via `utils/subgraph_method.py:subgraph_mining()`
6. Builds prototype bank, creates `MappedDataset` (VF2 pattern matching)
7. Trains `MyExplainerV2` or loads checkpoint, then evaluates

### Core Model (`models/myexplainerV2.py`)
`MyExplainerV2` architecture:
- **GCN Encoder**: 2-layer GCNConv for node embeddings
- **DeleteNet**: MLP predicting edge retention probabilities
- **AddVGAENet**: VGAE-style module generating candidate edges within frequent subgraph regions
- **Prototype alignment**: cosine similarity loss against class-specific subgraph embeddings

Loss components: CF prediction (cross-entropy + margin), L1 sparsity, VGAE reconstruction/KL, prototype alignment, oracle-guided delete ranking.

### Baseline Models (`models/`)
`atex_cf.py`, `c2explainer.py`, `cf_gnnexplainer.py`, `clear.py`, `rsgg_ce.py` — each has its own `__main__` block for standalone execution.

### GNN Classifiers (`gnns/`)
One classifier per dataset (e.g., `ba2motif_gnn.py` → `BA2MotifGCN`). Pre-trained weights stored in `param/gnns/`. All re-exported from `gnns/__init__.py`.

### Datasets (`datasets/`)
PyTorch Geometric `InMemoryDataset` wrappers. Factory function in `utils/dataset.py` maps dataset name to train/val/test splits.

### Key Utilities (`utils/`)
- `subgraph_method.py` — Frequent subgraph mining via igraph VF2, class-wise pattern generation
- `pair_data.py` — `MappedDataset` for VF2 pattern-to-graph matching, custom collate functions
- `train_myexplainer.py` — Training loop with validation, checkpointing, prototype refresh, LR scheduling
- `graph_utils.py` — SMARTS→PyG conversion (RDKit), explanatory subgraph extraction
- `loss_hparams.py` / `explainer_hparams.py` — Load per-dataset configs from YAML
- `simple_yaml.py` — Custom YAML parser (no PyYAML dependency)

### Graph Editor (`graph_editor/`)
Standalone web app for interactive graph editing and GNN probing. HTTP server with REST API (`/api/datasets`, `/api/graph`, `/api/predict`).

### Evaluation (`evaluationV2.py`)
Metrics: validity, proximity, fidelity, sparsity, runtime, oracle calls, per-class flip success. Supports `discrete` and `continuous` eval modes (`--eval_graph_mode`).

## Configuration

- `configs/loss_hparams.yaml` — Per-dataset loss weights (CF loss, L1, oracle, VGAE, prototype, margin)
- `configs/explainer_hparams.yaml` — Per-dataset explainer params (oracle_del_topk, etc.)
- `configs/node_label_config.yaml` — Atom/element labels for one-hot node features per dataset
- All configs use a custom YAML subset parser (`utils/simple_yaml.py`)

## Key Patterns

- Dataset-specific tuning is pervasive: loss weights, hyperparameters, and subgraph thresholds all vary per dataset
- Prototype bank is refreshed every N epochs during training (`--proto_refresh_every`)
- Oracle-guided delete ranking probes the frozen GNN to rank edge importance
- The `graph_editor/metadata.py` module is shared between the editor and the main pipeline for node label inference
