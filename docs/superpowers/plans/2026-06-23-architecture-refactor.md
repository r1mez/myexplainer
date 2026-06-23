# Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor CCFGExplainer's architecture to eliminate dead code, unify configuration, standardize GNN interfaces, consolidate evaluation metrics, and introduce a dataset registry — improving navigability, testability, and locality.

**Architecture:** The refactor proceeds in 6 tasks ordered by risk (lowest first). Each task is independently committable and leaves the codebase in a working state. The project has no formal test framework; verification is done by running the training pipeline end-to-end on a small dataset.

**Tech Stack:** Python 3.9.21, PyTorch 1.10.1+cu113, PyTorch Geometric 2.0.4

## Global Constraints

- No new dependencies may be added
- All changes must preserve existing CLI behavior (`python myexplainer_train_v2.py --dataset mutag ...`)
- Dataset-specific logic must remain per-dataset (no premature generalization)
- Git history preserves deleted code — never comment out when you can delete
- The `args` Namespace from argparse is the current god object; we replace it incrementally

---

## File Structure

### Files to DELETE (Task 1)
- `utils/param.py` — empty file
- `vis_debug.py` — broken (references undeclared `data`)
- `gnns/GNN.py` — unused GRETEL legacy
- `models/graph_conv.py` — duplicated in `myexplainerV2.py`

### Files to CLEAN (Task 1)
- `evaluationV2.py` — remove 800 lines of commented-out code
- `models/clear.py` — remove 700-line triple-quoted legacy string

### Files to CREATE
- `config/__init__.py` — package init
- `config/explainer_config.py` — `ExplainerConfig` dataclass (Task 2)
- `eval/__init__.py` — package init
- `eval/metrics.py` — consolidated evaluation metrics (Task 4)
- `utils/dataset_registry.py` — dataset registry dict (Task 5)

### Files to MODIFY
- `gnns/__init__.py` — remove `GNN.py` exports (Task 1)
- `myexplainer_train_v2.py` — use `ExplainerConfig`, registry (Tasks 2, 5)
- `models/myexplainerV2.py` — use `ExplainerConfig`, remove `DenseGATConv` duplicate (Tasks 1, 2, 3)
- `utils/train_myexplainer.py` — use `ExplainerConfig` (Task 2)
- `evaluationV2.py` — use `eval/metrics.py` (Task 4)
- `utils/baseline_eval_metrics.py` — delegate to `eval/metrics.py` (Task 4)
- `utils/subgraph_method.py` — type hints, remove `plt.show()`, use registry (Tasks 5, 6)
- `utils/pair_data.py` — type `PatternBank` parameter (Task 6)
- `gnns/ba2motif_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/mutag_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/bbbp_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/nci1_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/proteins_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/mutag188_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/benzene_gcn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/alkane_carbonyl_gnn.py` — unify `get_pred_explain` signature (Task 3)
- `gnns/fluoride_carbonyl_gnn.py` — unify `get_pred_explain` signature (Task 3)

---

### Task 1: Excise Dead Code

**Files:**
- Delete: `utils/param.py`, `vis_debug.py`, `gnns/GNN.py`, `models/graph_conv.py`
- Modify: `gnns/__init__.py:1-10`
- Modify: `evaluationV2.py:1-794` (delete commented-out block)
- Modify: `models/clear.py:1-711` (delete triple-quoted legacy)
- Modify: `models/myexplainerV2.py:12` (remove commented-out import)

**Interfaces:**
- Produces: clean files with only active code

- [ ] **Step 1: Delete empty and broken files**

```bash
git rm utils/param.py
git rm vis_debug.py
```

- [ ] **Step 2: Delete unused GNN.py and update gnns/__init__.py**

First, read `gnns/__init__.py` to understand current exports:

```bash
cat gnns/__init__.py
```

Remove any imports from `GNN.py`. The file currently re-exports all GNN classes. After removal, `gnns/__init__.py` should look like:

```python
from gnns.ba2motif_gnn import BA2MotifGCN
from gnns.mutag_gnn import Mutag_GCN
from gnns.bbbp_gnn import BBBP_GCN
from gnns.nci1_gnn import NCI1GCN
from gnns.proteins_gnn import PROTEINSGCN
from gnns.mutag188_gnn import Mutag188_GCN
from gnns.benzene_gcn import Benzene_GCN
from gnns.alkane_carbonyl_gnn import AlkaneCarbonylGCN
from gnns.fluoride_carbonyl_gnn import FluorideCarbonylGCN
```

```bash
git rm gnns/GNN.py
```

Edit `gnns/__init__.py` to remove the GNN.py import (if present). The `from gnns import *` in `myexplainer_train_v2.py` will still work because the per-dataset classes remain.

- [ ] **Step 3: Delete duplicated graph_conv.py**

```bash
git rm models/graph_conv.py
```

In `models/myexplainerV2.py`, line 12 has a commented-out import:
```python
# from graph_conv import DenseGATConv
```
Delete this line entirely. The `DenseGATConv` class defined in `myexplainerV2.py` (lines 359-405) is the one actually used.

- [ ] **Step 4: Clean evaluationV2.py — remove commented-out code**

The active code starts at line 798. Lines 1-794 are commented-out legacy. Replace the entire file with only the active code:

Read the active section (lines 798-end), then write it back as the entire file. The file should start with the docstring on line 799:

```python
"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from utils.batch_utils import core_data_from_batch, output_to_batch
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import torch.nn.functional as F

from utils.vis_utils import visualize_explainer_graph


def evaluate(args, model, gnn, data_loader):
    # ... (keep lines 823-915 exactly as they are)


def count_valid(target_lables, cf_graphs, gnn):
    # ... (keep lines 918-927 exactly as they are)


def compute_proximity(args, cf_graphs, ori_graphs):
    # ... (keep lines 998-1057 exactly as they are)


def compute_fidelity_prob(args, ori_graphs, cf_graphs, ori_prob, gnn):
    # ... (keep lines 1059-1088 exactly as they are)


def compute_sparsity(args, ori_graphs, cf_graphs):
    # ... (keep lines 1090-1101 exactly as they are)
```

Also remove the unused import on line 810:
```python
from networkx.classes import subgraph  # DELETE THIS LINE
```

- [ ] **Step 5: Clean models/clear.py — remove triple-quoted legacy**

Read the file to find where the active implementation starts (after the triple-quoted string ends). The active code begins after line ~711. Write back only the active code.

- [ ] **Step 6: Verify the pipeline still runs**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
```

Expected: training starts and completes 1 epoch without import errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: excise dead code and unused modules

- Delete utils/param.py (empty), vis_debug.py (broken)
- Delete gnns/GNN.py (unused GRETEL legacy)
- Delete models/graph_conv.py (duplicated in myexplainerV2.py)
- Remove 800 lines of commented-out code from evaluationV2.py
- Remove 700-line triple-quoted legacy from models/clear.py
- Remove commented-out import from myexplainerV2.py
"
```

---

### Task 2: Create ExplainerConfig Dataclass

**Files:**
- Create: `config/__init__.py`
- Create: `config/explainer_config.py`
- Modify: `myexplainer_train_v2.py`
- Modify: `models/myexplainerV2.py:287-337`
- Modify: `utils/train_myexplainer.py`

**Interfaces:**
- Produces: `ExplainerConfig` class with fields: `dataset`, `device`, `h_dim`, `z_dim`, `lr`, `weight_decay`, `epochs`, `batch_size`, `max_num_nodes`, `dropout`, `top_k`, `threshold`, `subgraph_method`, plus all loss hparams from YAML
- Consumes: `utils/loss_hparams.py:load_loss_hparams()`, `utils/explainer_hparams.py:load_explainer_hparams()`

- [ ] **Step 1: Create config package**

```bash
mkdir -p config
```

Create `config/__init__.py`:
```python
from config.explainer_config import ExplainerConfig
```

- [ ] **Step 2: Create ExplainerConfig dataclass**

Create `config/explainer_config.py`:

```python
"""Immutable configuration for MyExplainer pipeline.

Loads CLI args and YAML configs into a single validated object.
Replaces the mutable `args` Namespace that was threaded through every module.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from utils.loss_hparams import load_loss_hparams
from utils.explainer_hparams import load_explainer_hparams


@dataclass(frozen=True)
class ExplainerConfig:
    # --- Core settings ---
    dataset: str
    device: torch.device
    cuda: int = 0
    train_mode: bool = True
    task: str = "graph"

    # --- Data parameters ---
    top_k: int = 1
    threshold: float = 0.0
    batch_size: int = 256

    # --- Model parameters ---
    h_dim: int = 256
    z_dim: int = 32
    max_num_nodes: int = 25
    dropout: float = 0.1

    # --- Training parameters ---
    epochs: int = 1
    lr: float = 0.01
    weight_decay: float = 1e-5

    # --- Subgraph parameters ---
    subgraph_method: str = "genGraphEx"

    # --- GNN path ---
    gnn_path: str = "param/"

    # --- Loss hyperparameters (loaded from YAML) ---
    w_cf: float = 5.0
    w_l1_add: float = 0.1
    w_l1_del: float = 1.0
    w_oracle_del_rank: float = 1.0
    w_vgae_recon: float = 5.0
    w_vgae_kl: float = 1.0
    enable_fs_feature_recon: bool = False
    w_vgae_feat_recon: float = 1.0
    w_proto: float = 0.0
    cf_margin: float = 0.5
    lambda_cf_margin: float = 1.0

    # --- Dynamic fields (set after dataset load) ---
    x_dim: int = 0
    edge_attr_dim: int = 0

    @classmethod
    def from_args(cls, args) -> "ExplainerConfig":
        """Build config from argparse Namespace, loading YAML hparams."""
        device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

        # Load per-dataset loss hyperparameters from YAML
        loss_hparams = load_loss_hparams(args.dataset)

        # Load per-dataset explainer hyperparameters from YAML
        try:
            explainer_hparams = load_explainer_hparams(args.dataset)
        except (FileNotFoundError, KeyError):
            explainer_hparams = {}

        # Merge: CLI args override YAML defaults
        kw = dict(
            dataset=args.dataset,
            device=device,
            cuda=args.cuda,
            train_mode=getattr(args, "train_mode", True),
            task=getattr(args, "task", "graph"),
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            h_dim=args.h_dim,
            z_dim=args.z_dim,
            max_num_nodes=args.max_num_nodes,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            subgraph_method=args.subgraph_method,
            gnn_path=args.gnn_path,
        )
        # Apply YAML loss hparams
        kw.update(loss_hparams)
        # Apply YAML explainer hparams
        kw.update(explainer_hparams)

        return cls(**kw)

    def with_dataset_dims(self, x_dim: int, edge_attr_dim: int) -> "ExplainerConfig":
        """Return a new config with dataset-derived dimensions set."""
        # frozen=True means we need to create a new instance
        import dataclasses
        return dataclasses.replace(self, x_dim=x_dim, edge_attr_dim=edge_attr_dim)

    def with_device_override(self, device_str: str) -> "ExplainerConfig":
        """Return a new config with device override (for 'cpu' mode)."""
        import dataclasses
        device = torch.device(device_str)
        return dataclasses.replace(self, device=device)
```

- [ ] **Step 3: Update myexplainer_train_v2.py to use ExplainerConfig**

Replace the `main()` function. Key changes:
- Build `ExplainerConfig.from_args(args)` after parsing
- Use `config.device` instead of `args.device`
- Use `config.dataset` instead of `args.dataset`
- Pass `config` instead of `args` to all downstream functions

The updated `main()`:

```python
from config import ExplainerConfig


def main():
    args = parse_args()

    # Build immutable config from args + YAML
    config = ExplainerConfig.from_args(args)
    print(f"Using device: {config.device}")

    # Load dataset
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=config.dataset)
    x_dim = train_dataset[0].x.shape[1]
    edge_attr_dim = train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0
    config = config.with_dataset_dims(x_dim, edge_attr_dim)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load GNN
    print("\n2. Loading pre-trained GNN classifier...")
    gnn = torch.load(f'param/gnns/{config.dataset}_gcn.pt', map_location=config.device)
    for p in gnn.parameters():
        p.requires_grad_(False)
    gnn.eval()
    print("  GNN loaded successfully")

    # Predict on training set
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        for data in train_dataset:
            data = data.to(config.device)
            out = gnn(data.x, data.edge_index, data.batch)
            pred_probs.extend(out.softmax(dim=1))
            preds = out.argmax(dim=1).cpu()
            pred_labels.extend(preds)

    indices_0 = [i for i, pred in enumerate(pred_labels) if pred == 0]
    indices_1 = [i for i, pred in enumerate(pred_labels) if pred == 1]
    train_dataset_0, train_dataset_1 = train_dataset[indices_0], train_dataset[indices_1]
    splited_train_dataset = {0: train_dataset_0, 1: train_dataset_1}

    patterns = subgraph_mining(config, splited_train_dataset)

    print("\n4. Creating dataset with subgraph masks...")
    train_dataset_with_masks = MappedDataset(config, train_dataset, patterns, pred_labels, pred_probs)
    test_dataset_with_masks = MappedDataset(config, test_dataset, patterns, gnn=gnn)
    val_dataset_with_masks = MappedDataset(config, val_dataset, patterns, gnn=gnn)

    print("\n5. Creating masked data loader...")
    train_loader_masked = TorchDataLoader(
        train_dataset_with_masks, batch_size=config.batch_size, shuffle=False, collate_fn=train_collate_fn
    )
    test_loader_masked = TorchDataLoader(
        test_dataset_with_masks, batch_size=config.batch_size, shuffle=False, collate_fn=train_collate_fn
    )
    val_loader_masked = TorchDataLoader(
        val_dataset_with_masks, batch_size=config.batch_size, shuffle=False, collate_fn=train_collate_fn
    )
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total batches: {len(train_loader_masked)}")

    if config.train_mode:
        print("\n7. Initializing MyExplainer model...")
        model = MyExplainerV2(config, gnn).to(config.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {num_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, verbose=True, min_lr=1e-6
        )
        print(f"  Optimizer: Adam")
        print(f"  Learning rate: {config.lr}")
        print(f"  Weight decay: {config.weight_decay}")

        print("\n8. Training MyExplainer with subgraph masks...")
        print("=" * 80)

        trained_model, losses = train_myexplainerV2(
            config=config, model=model, gnn=gnn,
            train_loader=train_loader_masked, eval_loader=val_loader_masked,
            optimizer=optimizer, scheduler=scheduler, epochs=config.epochs
        )
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    else:
        print("\n8. Loading Trained MyExplainer...")
        print("=" * 80)
        trained_model = MyExplainerV2(config, gnn).to(config.device)
        trained_model.load_state_dict(torch.load(f'param/myexplainer_{config.dataset}_best.pt', map_location=config.device))
        trained_model.eval()
        for p in trained_model.parameters():
            p.requires_grad_(False)
        print("\n" + "=" * 80)
        print("Loading completed successfully!")
        print("=" * 80)

    print("\n9. Evaluating on validation set...")
    print("=" * 80)

    evaluation_metrics = evaluate(
        config=config, model=trained_model, gnn=gnn, data_loader=val_loader_masked,
    )
    # ... (print results using evaluation_metrics dict)
```

- [ ] **Step 4: Update MyExplainerV2 to accept ExplainerConfig**

In `models/myexplainerV2.py`, change the constructor and `compute_loss`:

```python
class MyExplainerV2(nn.Module):
    def __init__(self, config, gnn):
        super().__init__()
        self.config = config
        self.gnn = gnn
        self.device = config.device

        self.conv1 = GCNConv(config.x_dim, config.h_dim)
        self.conv2 = GCNConv(config.h_dim, config.h_dim)

        self.delete_net = DeleteNet(config.h_dim)
        self.add_net = AddVGAENet(config.h_dim, config.z_dim)

    # ... forward() stays the same ...

    def compute_loss(self, graphs, y_desired, outputs):
        """Compute loss using config hparams (no more args threading)."""
        cf_probs, cf_logits = self.gnn.get_pred_explain(
            graphs.x, outputs["edge_index_cf"], outputs["edge_weight_cf"], graphs.batch
        )

        y_target = y_desired.to(self.device).view(-1).long()
        cf_loss = self._compute_cf_loss(cf_logits, y_target)

        l1_add = outputs["p_add"].mean() if outputs["p_add"] is not None else 0.0
        l1_del = (1 - outputs["p_keep"]).mean()

        recon_loss = outputs["add_recon_loss"]
        kl_loss = outputs["add_kl_loss"]

        cfg = self.config
        total_loss = (
            cfg.w_cf * cf_loss +
            cfg.w_l1_add * l1_add +
            cfg.w_l1_del * l1_del +
            cfg.w_vgae_recon * recon_loss +
            cfg.w_vgae_kl * kl_loss
        )

        return {
            "total": total_loss,
            "cf": cf_loss.detach(),
            "recon": recon_loss.detach(),
            "kl": kl_loss.detach(),
        }

    def _compute_cf_loss(self, logits, y_target):
        """Compute CrossEntropy and Margin Loss using config."""
        ce_loss = F.cross_entropy(logits, y_target)

        B = logits.size(0)
        idx = torch.arange(B, device=self.device)
        logits_t = logits[idx, y_target]
        logits_o = logits[idx, 1 - y_target]

        cfg = self.config
        margin_loss = F.relu(cfg.cf_margin + logits_o - logits_t).mean()

        return ce_loss + cfg.lambda_cf_margin * margin_loss
```

- [ ] **Step 5: Update train_myexplainer.py to use config**

In `utils/train_myexplainer.py`, change function signature and internal calls:

```python
def train_myexplainerV2(config, model, gnn, train_loader, eval_loader, optimizer, scheduler, epochs=30):
    # ... (replace all `args.` references with `config.`)
    # In the training loop:
    #   loss_dict = model.compute_loss(origraphs, y_desired, outputs)
    #   (no more `args` parameter to compute_loss)
    # In checkpoint saving:
    #   torch.save(model.state_dict(), f'param/myexplainer_{config.dataset}_best.pt')
    # In validation:
    #   loss_dict = model.compute_loss(origraphs, y_desired, outputs)
```

- [ ] **Step 6: Update evaluate() to use config**

In `evaluationV2.py`, change function signature:

```python
def evaluate(config, model, gnn, data_loader):
    model.eval()
    gnn.eval()
    # Remove: args.train_mode = False (side effect on shared state!)
    # Remove: args.train_mode = True (at the end)
    # Use config.device instead of args.device
```

Also update all metric functions to accept `config` instead of `args`:
```python
def compute_proximity(config, cf_graphs, ori_graphs):
    # ... use config.device instead of args.device
```

- [ ] **Step 7: Update MappedDataset to use config**

In `utils/pair_data.py`, the constructor already accepts `args`. Change it to accept `config`:

```python
class MappedDataset(Dataset):
    def __init__(self, config, dataset, patterns, pred_labels=None, pred_probs=None, gnn=None):
        self.device = config.device
        self.dataset_name = config.dataset
        self.thresh = config.threshold
        self.config = config
        # ... rest stays the same, replace self.args with self.config
```

- [ ] **Step 8: Update subgraph_mining to use config**

In `utils/subgraph_method.py`, change function signature:

```python
def subgraph_mining(config, datasets):
    if config.subgraph_method == 'genGraphEx':
        if config.dataset == 'ba2motif':
            # ...
```

- [ ] **Step 9: Verify the pipeline still runs**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
```

Expected: training completes 1 epoch. Loss values should match pre-refactor.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: introduce ExplainerConfig dataclass

- Create config/explainer_config.py with frozen dataclass
- Loads YAML loss hparams and explainer hparams automatically
- Replace mutable args Namespace with immutable config
- Remove side-effect mutation of train_mode in evaluate()
- All modules now receive config instead of args
"
```

---

### Task 3: Unify GNN Classifier Interface

**Files:**
- Modify: `gnns/ba2motif_gnn.py:150-168`
- Modify: `gnns/mutag_gnn.py:99-108`
- Modify: `gnns/bbbp_gnn.py` (get_pred_explain)
- Modify: `gnns/nci1_gnn.py` (get_pred_explain)
- Modify: `gnns/proteins_gnn.py` (get_pred_explain)
- Modify: `gnns/mutag188_gnn.py` (get_pred_explain)
- Modify: `gnns/benzene_gcn.py` (get_pred_explain)
- Modify: `gnns/alkane_carbonyl_gnn.py:192-197`
- Modify: `gnns/fluoride_carbonyl_gnn.py` (get_pred_explain)
- Modify: `models/myexplainerV2.py:289-291`

**Interfaces:**
- Produces: unified `get_pred_explain(x, edge_index, edge_weight, batch)` on all GNNs
- Consumes: `MyExplainerV2.compute_loss` calls this interface

- [ ] **Step 1: Define the unified signature**

The target interface for all GNN classifiers:

```python
def get_pred_explain(self, x, edge_index, edge_weight, batch):
    """
    Explain interface: compute predictions with edge weights.
    
    Args:
        x: Node features [N, F]
        edge_index: Edge indices [2, E]
        edge_weight: Edge weights in [0, 1] range [E]
        batch: Batch vector [N]
    
    Returns:
        (probs, logits) tuple
    """
```

- [ ] **Step 2: Update BA2MotifGCN**

In `gnns/ba2motif_gnn.py`, replace the `get_pred_explain` method:

```python
def get_pred_explain(self, x, edge_index, edge_weight, batch):
    """Explain interface: compute predictions with edge weights in [0,1]."""
    graph_x = self.get_graph_rep(x, edge_index, batch, edge_weight=edge_weight)
    logits = self.classifier(graph_x)
    probs = self.softmax(logits)
    self.readout = probs
    return probs, logits
```

This removes the `mask_is_logit` parameter. The caller is responsible for providing weights in [0,1].

- [ ] **Step 3: Update Mutag_GCN**

In `gnns/mutag_gnn.py`, replace:

```python
def get_pred_explain(self, x, edge_index, edge_weight, batch):
    """Explain interface: compute predictions with edge weights in [0,1]."""
    for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
        x = conv(x, edge_index, edge_weight=edge_weight)
        x = relu(x)
    node_x = x
    graph_x = global_mean_pool(node_x, batch)
    pred = self.ffn(graph_x)
    self.readout = self.softmax(pred)
    return self.readout, pred
```

- [ ] **Step 4: Update remaining GNNs**

Apply the same pattern to each GNN. The key change: rename `edge_mask` parameter to `edge_weight`, ensure the weight is passed directly to the convolution layer (no internal sigmoid unless the GNN's architecture requires it for its own forward pass).

For each file, the pattern is:
1. Change parameter name from `edge_mask` to `edge_weight`
2. Pass `edge_weight` to the convolution's `edge_weight` parameter
3. Remove any internal `sigmoid` calls on the weight (caller provides [0,1])

Files to update:
- `gnns/bbbp_gnn.py`
- `gnns/nci1_gnn.py`
- `gnns/proteins_gnn.py`
- `gnns/mutag188_gnn.py`
- `gnns/benzene_gcn.py`
- `gnns/alkane_carbonyl_gnn.py`
- `gnns/fluoride_carbonyl_gnn.py`

- [ ] **Step 5: Update MyExplainerV2.compute_loss call site**

In `models/myexplainerV2.py`, the call in `compute_loss` already uses `edge_weight_cf` which is in [0,1]. No change needed to the call itself — the parameter rename from `edge_mask` to `edge_weight` in the GNNs makes the contract explicit.

- [ ] **Step 6: Verify the pipeline still runs**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
python myexplainer_train_v2.py --dataset mutag --epochs 1 --device cpu
```

Expected: both complete without errors. The GNN's behavior is unchanged because we're passing the same [0,1] weights — we just made the interface consistent.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: unify GNN get_pred_explain interface

- All 9 GNN classifiers now have identical signature:
  get_pred_explain(x, edge_index, edge_weight, batch)
- edge_weight is always in [0,1] range (caller's responsibility)
- Remove mask_is_logit parameter from BA2MotifGCN
- Remove internal sigmoid from BBBP_GCN
- Rename edge_mask to edge_weight for clarity
"
```

---

### Task 4: Consolidate Evaluation Metrics

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/metrics.py`
- Modify: `evaluationV2.py`
- Modify: `utils/baseline_eval_metrics.py`

**Interfaces:**
- Produces: `eval.metrics.proximity(config, cf_graphs, ori_graphs)`, `eval.metrics.fidelity(config, ori_graphs, cf_graphs, ori_prob, gnn)`, `eval.metrics.sparsity(config, ori_graphs, cf_graphs)`
- Consumes: `evaluationV2.evaluate()`, `utils/baseline_eval_metrics.py`

- [ ] **Step 1: Create eval package**

```bash
mkdir -p eval
```

Create `eval/__init__.py`:
```python
from eval.metrics import proximity, fidelity, sparsity
```

- [ ] **Step 2: Create eval/metrics.py**

Extract the metric functions from `evaluationV2.py`:

```python
"""Consolidated evaluation metrics for counterfactual explanations.

All metric functions accept either PyG Batch objects or work with the
config object for device resolution.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch

from utils.graph_utils import extract_explanatory_subgraph


def proximity(config, cf_graphs, ori_graphs):
    """Compute adjacency matrix distance (L1 Norm) between original and CF graphs.
    
    Args:
        config: ExplainerConfig with device attribute
        cf_graphs: Batch of counterfactual graphs
        ori_graphs: Batch of original graphs
    
    Returns:
        float: sum of per-graph proximity distances
    """
    rho = 1.0

    ori_list = ori_graphs.to_data_list()
    cf_list = cf_graphs.to_data_list()
    batch_size = len(ori_list)
    distances = torch.zeros(batch_size, device=config.device)

    for i in range(batch_size):
        orig_data = ori_list[i]
        cf_data = cf_list[i]

        # Determine unified node count N
        if getattr(orig_data, 'num_nodes', None) is not None:
            N = orig_data.num_nodes
        elif getattr(orig_data, 'x', None) is not None:
            N = orig_data.x.size(0)
        else:
            max_idx = 0
            if orig_data.edge_index.numel() > 0:
                max_idx = int(orig_data.edge_index.max())
            if cf_data.edge_index.numel() > 0:
                max_idx = max(max_idx, int(cf_data.edge_index.max()))
            N = max_idx + 1

        orig_adj = to_dense_adj(orig_data.edge_index, max_num_nodes=N).squeeze(0)
        cf_adj = to_dense_adj(cf_data.edge_index, max_num_nodes=N).squeeze(0)

        d_adj_entries = torch.norm(orig_adj - cf_adj, p=1)

        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges
        max_m = max(m_orig, m_cf)

        normalization = 2.0 * max_m if max_m > 0 else 1.0
        distances[i] = rho * (d_adj_entries / normalization)

    return distances.sum().item()


def fidelity(config, ori_graphs, cf_graphs, ori_prob, gnn):
    """Compute prediction probability drop for original class.
    
    Args:
        config: ExplainerConfig with device attribute
        ori_graphs: Batch of original graphs
        cf_graphs: Batch of counterfactual graphs
        ori_prob: Original prediction probabilities [N, num_classes]
        gnn: Pre-trained GNN classifier
    
    Returns:
        float: sum of per-graph fidelity values
    """
    ori_pred = ori_prob.argmax(dim=1)

    cf_pred_logits = gnn.get_pred(
        cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch
    )[0]
    cf_prob = F.softmax(cf_pred_logits, dim=1)

    fidelity_sum = 0.0
    for i in range(len(ori_pred)):
        ori_prob_single = ori_prob[i, ori_pred[i]].item()
        cf_prob_single = cf_prob[i, ori_pred[i]].item()
        fidelity_sum += (ori_prob_single - cf_prob_single)

    return fidelity_sum


def sparsity(config, ori_graphs, cf_graphs):
    """Compute 1 - (explanatory_edges / original_edges).
    
    Args:
        config: ExplainerConfig (unused but kept for interface consistency)
        ori_graphs: Batch of original graphs
        cf_graphs: Batch of counterfactual graphs
    
    Returns:
        float: sum of per-graph sparsity values
    """
    ori_list = ori_graphs.to_data_list()
    cf_list = cf_graphs.to_data_list()
    exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_list, cf_list)]

    sparsity_sum = 0.0
    for ori, exp in zip(ori_list, exp_graphs):
        ori_e = ori.num_edges
        exp_e = exp.num_edges
        sparsity_sum += 1 - (exp_e / ori_e) if ori_e > 0 else 0.0

    return sparsity_sum
```

- [ ] **Step 3: Update evaluationV2.py to use eval/metrics.py**

Replace the metric function definitions in `evaluationV2.py` with imports:

```python
from eval.metrics import proximity, fidelity, sparsity

# Remove the local compute_proximity, compute_fidelity_prob, compute_sparsity functions

def evaluate(config, model, gnn, data_loader):
    # ... (use proximity(config, ...), fidelity(config, ...), sparsity(config, ...))
```

- [ ] **Step 4: Update baseline_eval_metrics.py**

In `utils/baseline_eval_metrics.py`, keep the `OracleWrappedModel` class (it's baseline-specific), but delegate metric computation to `eval/metrics.py`:

```python
from eval.metrics import proximity, fidelity, sparsity

# Keep OracleWrappedModel
# Remove the local metric functions or make them thin wrappers
```

- [ ] **Step 5: Verify**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
```

Expected: evaluation runs, metrics print correctly.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: consolidate evaluation metrics into eval/metrics.py

- Create eval/metrics.py with proximity, fidelity, sparsity functions
- Remove duplicate implementations from evaluationV2.py
- baseline_eval_metrics.py delegates to eval/metrics.py
- All metric functions accept config instead of args
"
```

---

### Task 5: Dataset Registry

**Files:**
- Create: `utils/dataset_registry.py`
- Modify: `utils/dataset.py`
- Modify: `utils/subgraph_method.py`
- Modify: `myexplainer_train_v2.py` (GNN path resolution)

**Interfaces:**
- Produces: `DATASET_REGISTRY` dict, `get_dataset_entry(name)` function
- Consumes: `utils/dataset.py:get_datasets()`, `utils/subgraph_method.py:subgraph_mining()`

- [ ] **Step 1: Create dataset registry**

Create `utils/dataset_registry.py`:

```python
"""Central registry for dataset-specific configuration.

Each dataset has a single entry that captures:
- Dataset class and folder name
- GNN checkpoint path
- Subgraph mining parameters
- Split strategy
"""

from datasets import (
    Mutagenicity, MUTAG188, NCI1, AlkaneCarbonyl,
    FluorideCarbonyl, bbbp, BA2Motif, Benzene, PROTEINS,
)


# Type alias for the mining config
SubgraphMiningConfig = dict  # {"method": "continuous"|"discrete", "N": int, "num_samples": int}

DATASET_REGISTRY = {
    "mutag": {
        "cls": Mutagenicity,
        "folder": "mutag",
        "gnn_file": "mutag_gcn.pt",
        "subgraph": {"method": "discrete", "N": 417, "num_samples": 100},
        "split": "standard",  # train/eval/test modes
    },
    "mutag188": {
        "cls": MUTAG188,
        "folder": "mutag188",
        "gnn_file": "mutag188_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
    "nci1": {
        "cls": NCI1,
        "folder": "NCI1",
        "gnn_file": "nci1_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
    "bbbp": {
        "cls": bbbp,
        "folder": "bbbp",
        "gnn_file": "bbbp_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "slice",  # dataset[:200], [200:400], [400:]
    },
    "ba2motif": {
        "cls": BA2Motif,
        "folder": "ba2motif",
        "gnn_file": "ba2motif_gcn.pt",
        "subgraph": {"method": "continuous", "N": 25, "num_samples": 50},
        "split": "standard",
    },
    "benzene": {
        "cls": Benzene,
        "folder": "benzene",
        "gnn_file": "benzene_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
    "alkane_carbonyl": {
        "cls": AlkaneCarbonyl,
        "folder": "alkane_carbonyl",
        "gnn_file": "alkane_carbonyl_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
    "fluoride_carbonyl": {
        "cls": FluorideCarbonyl,
        "folder": "fluoride_carbonyl",
        "gnn_file": "fluoride_carbonyl_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
    "proteins": {
        "cls": PROTEINS,
        "folder": "proteins",
        "gnn_file": "proteins_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100},
        "split": "standard",
    },
}


def get_dataset_entry(name: str) -> dict:
    """Get registry entry for a dataset. Raises KeyError if not found."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[key]
```

- [ ] **Step 2: Update utils/dataset.py to use registry**

```python
import os
from utils.dataset_registry import get_dataset_entry

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def _resolve_dataset_root(root):
    if os.path.isabs(root):
        return root
    return os.path.abspath(os.path.join(PROJECT_ROOT, root))

def get_datasets(name, root="data/"):
    """Get preloaded datasets by name."""
    root = _resolve_dataset_root(root)
    entry = get_dataset_entry(name)
    folder = os.path.join(root, entry["folder"])
    cls = entry["cls"]

    print("Loading dataset: ", name)

    if entry["split"] == "slice":
        dataset = cls(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    else:
        train_dataset = cls(folder, mode="training")
        test_dataset = cls(folder, mode="testing")
        val_dataset = cls(folder, mode="evaluation")

    return train_dataset, val_dataset, test_dataset
```

- [ ] **Step 3: Update subgraph_mining to use registry**

In `utils/subgraph_method.py`:

```python
from utils.dataset_registry import get_dataset_entry

def subgraph_mining(config, datasets):
    entry = get_dataset_entry(config.dataset)
    sg_config = entry["subgraph"]
    method = sg_config["method"]
    N = sg_config["N"]
    num_samples = sg_config["num_samples"]

    if config.subgraph_method == 'genGraphEx':
        patterns_0 = []
        patterns_1 = []

        if method == "continuous":
            Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[0], N)
            for i in range(num_samples):
                patterns_0.append(graphsampler(N, Bdist, mean_estimate, result, Adj))

            Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[1], N)
            for i in range(num_samples):
                patterns_1.append(graphsampler(N, Bdist, mean_estimate, result, Adj))

        elif method == "discrete":
            X, Adj = GraphRepModelDiscrete(datasets[0], N)
            for i in range(num_samples):
                patterns_0.append(graphsamplerDiscrete(N, X, Adj))

            X, Adj = GraphRepModelDiscrete(datasets[1], N)
            for i in range(num_samples):
                patterns_1.append(graphsamplerDiscrete(N, X, Adj))

        else:
            raise ValueError(f"Unknown subgraph method: {method}")

        sort_key = lambda G: (G.number_of_nodes(), nx.density(G))
        patterns_0.sort(key=sort_key, reverse=True)
        patterns_1.sort(key=sort_key, reverse=True)
        return {0: patterns_0, 1: patterns_1}
```

- [ ] **Step 4: Update GNN path resolution in myexplainer_train_v2.py**

In `myexplainer_train_v2.py`, the GNN load line becomes:

```python
from utils.dataset_registry import get_dataset_entry

entry = get_dataset_entry(config.dataset)
gnn = torch.load(f'param/gnns/{entry["gnn_file"]}', map_location=config.device)
```

- [ ] **Step 5: Verify**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
python myexplainer_train_v2.py --dataset mutag --epochs 1 --device cpu
```

Expected: both datasets load and train correctly using registry-based config.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: introduce dataset registry

- Create utils/dataset_registry.py with DATASET_REGISTRY dict
- Each dataset has one entry: class, folder, gnn_file, subgraph config
- utils/dataset.py uses registry instead of if/elif chain
- utils/subgraph_method.py uses registry instead of hardcoded N values
- Adding a new dataset = adding one registry entry
"
```

---

### Task 6: Clean Up Subgraph Mining Module

**Files:**
- Modify: `utils/subgraph_method.py`
- Modify: `utils/pair_data.py`

**Interfaces:**
- Produces: `PatternBank = Dict[int, List[nx.Graph]]` type alias
- Consumes: `MappedDataset.__init__` type-hinted parameter

- [ ] **Step 1: Remove plt.show() calls from subgraph_method.py**

In `utils/subgraph_method.py`, find and remove the `plt.show()` calls:

In `graphsampler()` (around line 210-215), remove:
```python
    # DELETE THESE LINES:
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos=pos, node_size=50, node_color='red',
                     edge_color='gray', with_labels=True, width=1.5)
    plt.title(f"Generated Graph (Threshold={threshold})")
    plt.show()
```

In `graphsamplerDiscrete()` (around line 357-367), remove:
```python
    # DELETE THESE LINES:
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6', edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='sans-serif')
    plt.title(f"Generated Graph (Threshold={threshold})", fontsize=15)
    plt.axis('off')
    plt.show()
```

Also remove the unused `import matplotlib.pyplot as plt` if no other usage remains.

- [ ] **Step 2: Add PatternBank type alias and error handling**

At the top of `utils/subgraph_method.py`, add:

```python
from typing import Dict, List
import networkx as nx

PatternBank = Dict[int, List[nx.Graph]]
```

Update the function signature and add validation:

```python
def subgraph_mining(config, datasets) -> PatternBank:
    """Mine frequent subgraph patterns per class.
    
    Args:
        config: ExplainerConfig with dataset and subgraph_method
        datasets: dict {0: class_0_data, 1: class_1_data}
    
    Returns:
        PatternBank: {0: [patterns for class 0], 1: [patterns for class 1]}
    
    Raises:
        KeyError: if dataset not in registry
        ValueError: if subgraph method unknown
    """
    entry = get_dataset_entry(config.dataset)
    # ... (rest of implementation from Task 5)
```

- [ ] **Step 3: Type the MappedDataset constructor**

In `utils/pair_data.py`, add the type hint:

```python
from utils.subgraph_method import PatternBank

class MappedDataset(Dataset):
    def __init__(self, config, dataset, patterns: PatternBank, 
                 pred_labels=None, pred_probs=None, gnn=None):
        self.patterns_0 = patterns[0]  # List[nx.Graph]
        self.patterns_1 = patterns[1]  # List[nx.Graph]
        # ... rest stays the same
```

- [ ] **Step 4: Verify**

```bash
python myexplainer_train_v2.py --dataset ba2motif --epochs 1 --device cpu
```

Expected: no matplotlib windows pop up, training completes normally.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: clean up subgraph mining module

- Remove plt.show() calls that block headless servers
- Add PatternBank type alias for pattern dict
- Type-hint MappedDataset constructor
- Add error handling for unknown datasets
"
```

---

### Task 7: Extract Baseline Evaluation Boilerplate (Optional)

**Files:**
- Create: `eval/baseline_runner.py`
- Modify: `models/atex_cf.py`, `models/c2explainer.py`, `models/cf_gnnexplainer.py`, `models/clear.py`, `models/rsgg_ce.py`

**Interfaces:**
- Produces: `BaselineRunner` class with `run(data_loader)` and `visualize(...)` methods
- Consumes: each baseline model's interface

- [ ] **Step 1: Identify shared patterns**

Read the `evaluate_*` and `visualize_comparison` functions in each baseline. Note the common structure:
1. Loop over data_loader
2. Run model to get CF graphs
3. Compute metrics (validity, proximity, fidelity, sparsity)
4. Optionally visualize

- [ ] **Step 2: Create BaselineRunner**

Create `eval/baseline_runner.py`:

```python
"""Shared evaluation harness for baseline counterfactual explainer models."""
import torch
from tqdm import tqdm
from eval.metrics import proximity, fidelity, sparsity


class BaselineRunner:
    """Run evaluation for any baseline counterfactual explainer.
    
    The baseline model must implement:
        - model(data) -> cf_graphs (or similar interface)
    """
    
    def __init__(self, model, gnn, config):
        self.model = model
        self.gnn = gnn
        self.config = config
    
    def run(self, data_loader, model_forward_fn=None):
        """Run evaluation and return metrics dict.
        
        Args:
            data_loader: DataLoader for evaluation data
            model_forward_fn: callable(model, batch) -> cf_graphs
                             If None, uses model(batch)
        
        Returns:
            dict with validity, proximity, fidelity, sparsity
        """
        if model_forward_fn is None:
            model_forward_fn = lambda m, b: m(b)
        
        self.model.eval()
        self.gnn.eval()
        
        results = {"valid": 0, "total": 0, "proximity": 0.0, "fidelity": 0.0, "sparsity": 0.0}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating baseline"):
                # ... common evaluation logic
                pass
        
        return results
```

- [ ] **Step 3: Refactor one baseline as proof of concept**

Pick `models/c2explainer.py` as the first baseline to refactor. Replace its `evaluate_c2` function with a `BaselineRunner` call.

- [ ] **Step 4: Verify**

```bash
python -m models.c2explainer
```

Expected: baseline evaluation produces same results.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: extract baseline evaluation boilerplate

- Create eval/baseline_runner.py with shared evaluation harness
- Refactor c2explainer.py as proof of concept
- Other baselines can be migrated incrementally
"
```

---

## Verification Checklist

After all tasks are complete, verify the full pipeline:

```bash
# Test main pipeline on multiple datasets
python myexplainer_train_v2.py --dataset ba2motif --epochs 2 --device cpu
python myexplainer_train_v2.py --dataset mutag --epochs 2 --device cpu

# Test evaluation mode
python myexplainer_train_v2.py --dataset ba2motif --train_mode False --device cpu

# Verify no import errors
python -c "from config import ExplainerConfig; print('OK')"
python -c "from eval.metrics import proximity, fidelity, sparsity; print('OK')"
python -c "from utils.dataset_registry import DATASET_REGISTRY; print(f'Registered: {list(DATASET_REGISTRY.keys())}')"
```
