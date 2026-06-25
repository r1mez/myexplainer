# Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the CCFGExplainer codebase to improve modularity, reduce duplication, and make experiments easier to run — one commit per architectural change.

**Architecture:** The refactor proceeds in 8 self-contained tasks ordered by dependency. Each task produces a working codebase. The core changes: extract a GNN base class, define a BaseExplainer interface, consolidate configuration, and split grab-bag modules.

**Tech Stack:** Python 3.9, PyTorch 1.10.1, PyTorch Geometric 2.0.4

## Global Constraints

- Binary classification only (`num_classes = 2`) throughout
- Seed fixed to 42 via `set_seed(42)`
- No formal test framework exists — verify via `python myexplainer_train_v2.py --dataset ba2motif --train_mode False --device cpu` (evaluation-only mode)
- Remote server runs conda env `myexplainer`, Python 3.9.21
- GNN checkpoints stored at `param/gnns/{dataset}_gcn.pt`
- All GNN classes expose: `forward`, `get_pred`, `get_pred_explain`, `get_node_reps`, `get_graph_rep`

---

## File Structure (final state after all tasks)

```
gnns/
  __init__.py                    # modified: exports BaseGNNClassifier
  base.py                        # NEW: BaseGNNClassifier abstract base
  ba2motif_gnn.py                # modified: extends BaseGNNClassifier
  mutag_gnn.py                   # modified: extends BaseGNNClassifier
  bbbp_gnn.py                    # modified: extends BaseGNNClassifier
  nci1_gnn.py                    # modified: extends BaseGNNClassifier
  benzene_gcn.py                 # modified: extends BaseGNNClassifier
  proteins_gnn.py                # modified: extends BaseGNNClassifier
  mutag188_gnn.py                # modified: extends BaseGNNClassifier
  alkane_carbonyl_gnn.py         # modified: extends BaseGNNClassifier
  fluoride_carbonyl_gnn.py       # modified: extends BaseGNNClassifier
  model_utils/                   # unchanged
models/
  __init__.py                    # NEW: package init
  base.py                        # NEW: BaseExplainer + CFResult
  myexplainerV2.py               # modified: extends BaseExplainer
  atex_cf.py                     # modified: extends BaseExplainer, eval removed
  c2explainer.py                 # modified: extends BaseExplainer, eval removed
  cf_gnnexplainer.py             # modified: extends BaseExplainer, eval removed
  clear.py                       # modified: extends BaseExplainer, eval removed
  rsgg_ce.py                     # modified: extends BaseExplainer, eval removed
utils/
  graph_utils.py                 # modified: chemistry + output funcs removed
  chemistry.py                   # NEW: atom/bond maps, smarts_to_data, data_to_mol
  graph_ops.py                   # NEW: extract_explanatory_subgraph, exclude
  output_conversion.py           # NEW: process_outputs, output_to_batch
  node_labels.py                 # NEW: infer labels (extracted from graph_editor/metadata)
  vis_utils.py                   # modified: imports from node_labels
  loss_hparams.py                # removed (merged into config)
  explainer_hparams.py           # removed (merged into config)
  pair_data.py                   # modified: dead code removed
  train_myexplainer.py           # modified: vis behind flag
  ...
eval/
  baseline_runner.py             # modified: uses BaseExplainer interface
  metrics.py                     # unchanged
evaluationV2.py                  # modified: vis behind flag, imports from graph_ops
graph_editor/
  metadata.py                    # modified: imports from node_labels
myexplainer_train_v2.py          # unchanged in Tasks 1-7
```

---

## Task 1: Remove dead code and debugging artifacts

**Files:**
- Modify: `utils/pair_data.py` — delete `custom_collate_fn`
- Modify: `utils/loss_hparams.py` — delete `apply_loss_hparams`
- Modify: `utils/explainer_hparams.py` — delete `apply_explainer_hparams`
- Modify: `utils/train_myexplainer.py` — guard `visualize_explainer_graph` behind flag
- Modify: `evaluationV2.py` — guard `visualize_explainer_graph` behind flag
- Modify: `models/cf_gnnexplainer.py` — delete commented-out evaluation block
- Modify: `utils/__init__.py` — remove `custom_collate_fn` export if present

**Interfaces:**
- Consumes: nothing (pure deletion)
- Produces: cleaner codebase, no behavioral change

- [ ] **Step 1: Delete `custom_collate_fn` from `utils/pair_data.py`**

Read `utils/pair_data.py`. Delete the `custom_collate_fn` function (the one handling `ori_graph`/`tgt_graph` format — NOT `train_collate_fn`). It is unused in the V2 pipeline.

- [ ] **Step 2: Delete `apply_loss_hparams` from `utils/loss_hparams.py`**

Read `utils/loss_hparams.py`. Delete the `apply_loss_hparams` function (lines ~86-94). It mutates an argparse namespace and is never called in the V2 pipeline. Keep `load_loss_hparams`.

- [ ] **Step 3: Delete `apply_explainer_hparams` from `utils/explainer_hparams.py`**

Read `utils/explainer_hparams.py`. Delete the `apply_explainer_hparams` function (lines ~74-83). Keep `load_explainer_hparams`.

- [ ] **Step 4: Guard visualization in `utils/train_myexplainer.py`**

Read `utils/train_myexplainer.py`. Find the `visualize_explainer_graph` call(s) inside the training loop. Wrap them in `if getattr(config, 'visualize', False):` so they only run when explicitly requested. The import at the top can stay.

- [ ] **Step 5: Guard visualization in `evaluationV2.py`**

Read `evaluationV2.py`. Find the `visualize_explainer_graph` call(s). Wrap in `if getattr(config, 'visualize', False):`. Also replace `data_loader.dataset.__len__()` with `len(data_loader.dataset)`.

- [ ] **Step 6: Delete commented-out code in `models/cf_gnnexplainer.py`**

Read `models/cf_gnnexplainer.py`. Find the large block of commented-out code (~150 lines of old evaluation function). Delete it entirely.

- [ ] **Step 7: Remove `custom_collate_fn` from `utils/__init__.py`**

Read `utils/__init__.py`. If `custom_collate_fn` is exported, remove it from the import/`__all__`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: remove dead code and guard debugging artifacts

- Delete unused custom_collate_fn (V1 leftover)
- Delete apply_loss_hparams / apply_explainer_hparams (never called in V2)
- Guard visualize_explainer_graph behind config.visualize flag
- Delete ~150 lines commented-out code in cf_gnnexplainer.py
- Fix __len__() dunder call to len()"
```

---

## Task 2: Fix reverse dependency — extract `node_labels.py`

**Files:**
- Create: `utils/node_labels.py` — node label inference logic
- Modify: `utils/vis_utils.py` — import from `node_labels` instead of `graph_editor`
- Modify: `graph_editor/metadata.py` — import from `node_labels` instead of duplicating

**Interfaces:**
- Consumes: `utils/simple_yaml.load_yaml_file`
- Produces:
  - `utils/node_labels.infer_feature_mode(dataset_name, data) -> str`
  - `utils/node_labels.infer_node_label(dataset_name, data, node_idx) -> str`
  - `utils/node_labels.infer_node_labels_for_dataset(dataset_name, data) -> List[str]`
  - `utils/node_labels.feature_labels_for_dataset(dataset_name) -> List[str]`
  - `utils/node_labels.node_label_mode_for_dataset(dataset_name) -> str`

- [ ] **Step 1: Read both files to understand what to extract**

Read `graph_editor/metadata.py` fully. Identify the functions: `infer_feature_mode`, `infer_node_label`, `infer_node_labels_for_dataset`, `feature_labels_for_dataset`, `node_label_mode_for_dataset`. Read `utils/vis_utils.py` to see which of these it imports.

- [ ] **Step 2: Create `utils/node_labels.py`**

Create the file with the extracted functions. These functions should be self-contained — they depend only on `utils/simple_yaml` and the `configs/node_label_config.yaml` file.

```python
"""Node label inference for graph visualization."""
import os
from typing import List
from utils.simple_yaml import load_yaml_file

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "node_label_config.yaml")


def _load_dataset_config(dataset_name: str) -> dict:
    cfg = load_yaml_file(_CONFIG_PATH)
    if dataset_name not in cfg:
        raise ValueError(f"No node label config for dataset '{dataset_name}'")
    return cfg[dataset_name]


def infer_feature_mode(dataset_name: str, data) -> str:
    cfg = _load_dataset_config(dataset_name)
    if "mode" in cfg:
        return cfg["mode"]
    return "onehot" if data.x.size(1) <= 20 else "vector"


def node_label_mode_for_dataset(dataset_name: str) -> str:
    cfg = _load_dataset_config(dataset_name)
    return cfg.get("mode", "onehot")


def feature_labels_for_dataset(dataset_name: str) -> List[str]:
    cfg = _load_dataset_config(dataset_name)
    return cfg.get("labels", [])


def infer_node_label(dataset_name: str, data, node_idx: int) -> str:
    mode = node_label_mode_for_dataset(dataset_name)
    labels = feature_labels_for_dataset(dataset_name)
    x = data.x[node_idx]
    if mode == "onehot" and labels:
        idx = int(x.argmax().item())
        return labels[idx] if idx < len(labels) else f"?{idx}"
    elif mode == "atomic_num":
        return str(int(x[0].item()))
    else:
        return f"n{node_idx}"


def infer_node_labels_for_dataset(dataset_name: str, data) -> List[str]:
    return [infer_node_label(dataset_name, data, i) for i in range(data.x.size(0))]
```

- [ ] **Step 3: Update `utils/vis_utils.py`**

Read `utils/vis_utils.py`. Change imports from `graph_editor.metadata` to `utils.node_labels`.

- [ ] **Step 4: Update `graph_editor/metadata.py`**

Read `graph_editor/metadata.py`. Replace the implementation of the 5 functions with re-exports from `utils.node_labels`. Keep `resolve_model_path` and `SUPPORTED_DATASETS` in place.

- [ ] **Step 5: Verify no circular imports**

```bash
python -c "from utils.node_labels import infer_feature_mode; print('OK')"
python -c "from graph_editor.metadata import infer_feature_mode; print('OK')"
python -c "from utils.vis_utils import visualize_explainer_graph; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract node_labels.py to fix reverse dependency

- utils/vis_utils no longer imports from graph_editor
- graph_editor/metadata re-exports from utils/node_labels
- Node label inference logic shared via single module"
```

---

## Task 3: Extract GNN base class

**Files:**
- Create: `gnns/base.py` — `BaseGNNClassifier(nn.Module)`
- Modify: `gnns/ba2motif_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/mutag_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/bbbp_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/nci1_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/benzene_gcn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/proteins_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/mutag188_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/alkane_carbonyl_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/fluoride_carbonyl_gnn.py` — extend `BaseGNNClassifier`
- Modify: `gnns/__init__.py` — export `BaseGNNClassifier`

**Interfaces:**
- Consumes: `torch.nn`, `torch_geometric.nn`
- Produces: `BaseGNNClassifier` with:
  - `_forward_convs(self, x, edge_index, edge_weight=None) -> Tensor` — abstract
  - `_pool(self, node_emb, batch) -> Tensor` — abstract
  - `_classify(self, graph_emb) -> Tensor` — abstract
  - `forward(self, x, edge_index, batch) -> Tensor`
  - `get_node_reps(self, x, edge_index, edge_weight=None) -> Tensor`
  - `get_graph_rep(self, x, edge_index, batch, edge_weight=None) -> Tensor`
  - `get_pred(self, x, edge_index, batch) -> tuple[Tensor, Tensor]`
  - `get_pred_explain(self, x, edge_index, edge_weight, batch) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: Create `gnns/base.py`**

```python
"""Base class for all GNN classifiers in the explainer pipeline."""
from abc import abstractmethod
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor


class BaseGNNClassifier(torch.nn.Module):
    """Abstract base for GNN graph classifiers.

    Subclasses implement:
        - _forward_convs(x, edge_index, edge_weight): node embeddings
        - _pool(node_emb, batch): graph embeddings
        - _classify(graph_emb): logits
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    @abstractmethod
    def _forward_convs(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        ...

    @abstractmethod
    def _pool(self, node_emb: Tensor, batch: Tensor) -> Tensor:
        ...

    @abstractmethod
    def _classify(self, graph_emb: Tensor) -> Tensor:
        ...

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        node_emb = self._forward_convs(x, edge_index)
        graph_emb = self._pool(node_emb, batch)
        return self._classify(graph_emb)

    def get_node_reps(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        return self._forward_convs(x, edge_index, edge_weight)

    def get_graph_rep(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_weight: Tensor = None) -> Tensor:
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        return self._pool(node_emb, batch)

    def get_pred(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.forward(x, edge_index, batch)
        return F.softmax(logits, dim=-1), logits

    def get_pred_explain(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        graph_emb = self._pool(node_emb, batch)
        logits = self._classify(graph_emb)
        return F.softmax(logits, dim=-1), logits
```

- [ ] **Step 2: Migrate `gnns/ba2motif_gnn.py`**

Read the file. Change `class BA2MotifGCN(torch.nn.Module)` to `class BA2MotifGCN(BaseGNNClassifier)`. Replace the 5 duplicated methods by implementing the 3 abstract methods (`_forward_convs`, `_pool`, `_classify`). Keep `__init__` and custom methods. Add `from gnns.base import BaseGNNClassifier`.

- [ ] **Step 3: Migrate `gnns/mutag_gnn.py`**

Same pattern. Note: `in_channels` hardcoded to 14.

- [ ] **Step 4: Migrate `gnns/bbbp_gnn.py`**

Same pattern. Keep extra `get_emb` method. `in_channels` hardcoded to 9.

- [ ] **Step 5: Migrate `gnns/nci1_gnn.py`**

Same pattern. Uses `LEConv`. `in_channels` hardcoded to 37.

- [ ] **Step 6: Migrate `gnns/benzene_gcn.py`**

Same pattern. `in_channels` hardcoded to 14.

- [ ] **Step 7: Migrate `gnns/proteins_gnn.py`**

Same pattern. Uses `global_max_pool`.

- [ ] **Step 8: Migrate `gnns/mutag188_gnn.py`**

Same pattern.

- [ ] **Step 9: Migrate `gnns/alkane_carbonyl_gnn.py`**

Same pattern. Uses `EdgeWeightedGATConv`, residual connections, configurable pooling.

- [ ] **Step 10: Migrate `gnns/fluoride_carbonyl_gnn.py`**

Same as alkane_carbonyl.

- [ ] **Step 11: Update `gnns/__init__.py`**

Add `from gnns.base import BaseGNNClassifier` to exports.

- [ ] **Step 12: Verify all GNNs still work**

```bash
python -c "
from gnns import *
print('All GNN classes imported successfully')
"
```

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: extract BaseGNNClassifier base class

- All 9 GNN classes now extend BaseGNNClassifier
- 5 interface methods implemented once in base class
- Subclasses implement 3 abstract methods: _forward_convs, _pool, _classify
- ~450 lines of duplicated method code eliminated"
```

---

## Task 4: Split `utils/graph_utils.py` into focused modules

**Files:**
- Create: `utils/chemistry.py` — atom/bond maps, `smarts_to_data`, `data_to_mol`
- Create: `utils/graph_ops.py` — `extract_explanatory_subgraph`, `exclude_explanatory_subgraph`
- Create: `utils/output_conversion.py` — `process_outputs`, `output_to_batch`
- Modify: `utils/graph_utils.py` — keep as thin re-export layer
- Modify: `utils/batch_utils.py` — move `output_to_batch` to `output_conversion.py`
- Modify: `eval/metrics.py` — import from `graph_ops`
- Modify: `evaluationV2.py` — import from `graph_ops` and `output_conversion`
- Modify: `utils/train_utils.py` — import from `graph_ops`
- Modify: `models/myexplainerV2.py` — import from `output_conversion`

- [ ] **Step 1: Create `utils/chemistry.py`**

Read `utils/graph_utils.py`. Extract all atom/bond map dicts, `smarts_to_data`, `data_to_mol`, and RDKit-dependent functions.

- [ ] **Step 2: Create `utils/graph_ops.py`**

Extract `extract_explanatory_subgraph` and `exclude_explanatory_subgraph`. Deduplicate the shared `normalize_input` helper.

- [ ] **Step 3: Create `utils/output_conversion.py`**

Extract `process_outputs` from `graph_utils.py` and `output_to_batch` from `batch_utils.py`.

- [ ] **Step 4: Update `utils/graph_utils.py` as re-export layer**

```python
"""Backward-compatible re-exports. Prefer importing from chemistry, graph_ops, output_conversion."""
from utils.chemistry import *  # noqa
from utils.graph_ops import extract_explanatory_subgraph, exclude_explanatory_subgraph  # noqa
from utils.output_conversion import process_outputs  # noqa
```

- [ ] **Step 5: Update `utils/batch_utils.py`**

Remove `output_to_batch` (moved). Add re-export from `output_conversion`.

- [ ] **Step 6: Update internal imports**

- `eval/metrics.py`: `from utils.graph_ops import extract_explanatory_subgraph`
- `evaluationV2.py`: `from utils.graph_ops import ...` and `from utils.output_conversion import output_to_batch`
- `utils/train_utils.py`: `from utils.graph_ops import ...`
- `models/myexplainerV2.py`: `from utils.output_conversion import process_outputs`

- [ ] **Step 7: Verify imports**

```bash
python -c "
from utils.chemistry import smarts_to_data
from utils.graph_ops import extract_explanatory_subgraph
from utils.output_conversion import process_outputs, output_to_batch
from utils.graph_utils import extract_explanatory_subgraph  # backward compat
print('All imports OK')
"
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: split graph_utils.py into focused modules

- utils/chemistry.py: atom/bond maps, RDKit conversions
- utils/graph_ops.py: extract_explanatory_subgraph, exclude (pure graph ops)
- utils/output_conversion.py: process_outputs, output_to_batch
- graph_utils.py kept as re-export layer for backward compat
- Deduplicated normalize_input helper"
```

---

## Task 5: Define BaseExplainer interface and unify evaluation

**Files:**
- Create: `models/__init__.py`
- Create: `models/base.py` — `BaseExplainer`, `CFResult`
- Modify: `models/atex_cf.py` — extend `BaseExplainer`, delete eval function
- Modify: `models/c2explainer.py` — extend `BaseExplainer`, delete eval function
- Modify: `models/cf_gnnexplainer.py` — extend `BaseExplainer`, delete eval function
- Modify: `models/clear.py` — extend `BaseExplainer`, delete eval function
- Modify: `models/rsgg_ce.py` — extend `BaseExplainer`, delete eval function
- Modify: `eval/baseline_runner.py` — use `BaseExplainer.explain_graph()` + `CFResult`

- [ ] **Step 1: Create `models/__init__.py`**

```python
"""Counterfactual graph explainer models."""
```

- [ ] **Step 2: Create `models/base.py`**

```python
"""Base interface for counterfactual graph explainers."""
from dataclasses import dataclass
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class CFResult:
    cf_edge_index: Tensor       # [2, E_cf]
    cf_edge_weight: Tensor      # [E_cf]
    oracle_calls: int = 0
    runtime: float = 0.0


class BaseExplainer(nn.Module):
    def explain_graph(self, data: Data, device: str = "cpu") -> CFResult:
        raise NotImplementedError

    def fit(self, train_dataset, gnn, device: str = "cpu") -> None:
        pass
```

- [ ] **Step 3: Migrate `models/atex_cf.py`**

Add `from models.base import BaseExplainer, CFResult`. Change class to extend `BaseExplainer`. Implement `explain_graph` returning `CFResult`. Delete `evaluate_atex_cf_graph`. Update `__main__` to use `BaselineRunner`.

- [ ] **Step 4: Migrate `models/c2explainer.py`**

Same pattern. Delete `evaluate_c2_structural`. Delete duplicate `visualize_comparison`.

- [ ] **Step 5: Migrate `models/cf_gnnexplainer.py`**

Same. `run_one_graph` returns dense adj — `explain_graph` converts to sparse edge_index.

- [ ] **Step 6: Migrate `models/clear.py`**

Same. `fit()` calls `train_graphcfe()`. `explain_graph` wraps `_generate_single_cf`.

- [ ] **Step 7: Migrate `models/rsgg_ce.py`**

Same. `fit()` calls `train_rsgg_ce()`.

- [ ] **Step 8: Update `eval/baseline_runner.py`**

Update `run` to accept `BaseExplainer`, call `explainer.explain_graph(data, device)`, build CF graph from `CFResult`, compute metrics.

- [ ] **Step 9: Update `__main__` blocks in baselines**

Each `__main__` block uses `BaselineRunner` instead of its own eval function.

- [ ] **Step 10: Verify**

```bash
python -c "
from models.base import BaseExplainer, CFResult
from models.atex_cf import ATEXCFExplainer
from models.c2explainer import C2ExplainerStructuralOnly
print('All models import BaseExplainer')
"
```

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: define BaseExplainer interface and unify evaluation

- models/base.py: BaseExplainer with explain_graph() -> CFResult
- All 6 explainer models extend BaseExplainer
- CFResult: cf_edge_index, cf_edge_weight, oracle_calls, runtime
- Deleted ~580 lines of duplicated evaluation functions
- BaselineRunner updated to use BaseExplainer interface"
```

---

## Task 6: Consolidate configuration

**Files:**
- Modify: `config/explainer_config.py` — absorb hardcoded values
- Modify: `myexplainer_train_v2.py` — read all params from config
- Modify: `utils/train_myexplainer.py` — read grad clip, checkpoint interval from config
- Remove: `utils/loss_hparams.py` — merged into config
- Remove: `utils/explainer_hparams.py` — merged into config

- [ ] **Step 1: Read current `config/explainer_config.py`**

- [ ] **Step 2: Add new fields to `ExplainerConfig`**

```python
optimizer: str = "adam"
lr: float = 0.001
scheduler_factor: float = 0.8
scheduler_patience: int = 15
scheduler_min_lr: float = 1e-6
grad_clip_max_norm: float = 1.0
checkpoint_interval: int = 10
batch_size: int = 64
```

- [ ] **Step 3: Update `myexplainer_train_v2.py`**

Replace hardcoded values with `config.*`.

- [ ] **Step 4: Update `utils/train_myexplainer.py`**

Replace hardcoded `max_norm=1.0` with `config.grad_clip_max_norm`. Replace checkpoint interval.

- [ ] **Step 5: Remove `utils/loss_hparams.py` and `utils/explainer_hparams.py`**

Merge YAML loading into `ExplainerConfig.from_args()`.

- [ ] **Step 6: Verify**

```bash
python -c "
from config import ExplainerConfig
import argparse
args = argparse.Namespace(dataset='ba2motif', cuda='0', device='cpu', epochs=100, lr=0.001, h_dim=64, z_dim=32, batch_size=64, train_mode='True')
config = ExplainerConfig.from_args(args)
print(f'Config: lr={config.lr}, grad_clip={config.grad_clip_max_norm}')
"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: consolidate configuration into ExplainerConfig

- All training params now in ExplainerConfig with defaults
- Removed loss_hparams.py and explainer_hparams.py
- No more hardcoded values scattered across modules"
```

---

## Task 7: Consolidate dataset onboarding

**Files:**
- Modify: `utils/dataset_registry.py` — auto-discovery
- Modify: `utils/chemistry.py` — atom maps from YAML

- [ ] **Step 1: Extend `configs/node_label_config.yaml` with atom/bond maps**

- [ ] **Step 2: Update `utils/dataset_registry.py` for auto-discovery**

- [ ] **Step 3: Update `utils/chemistry.py` to load from YAML**

- [ ] **Step 4: Verify**

```bash
python -c "
from utils.dataset_registry import get_dataset_entry
for name in ['mutag', 'ba2motif', 'bbbp']:
    print(f'{name}: {get_dataset_entry(name)[\"gnn_file\"]}')
"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: consolidate dataset onboarding

- Atom/bond maps moved to YAML config
- Dataset registry auto-discovers from configs
- Adding a dataset: 3 files (dataset.py + gnn.py + config.yaml)"
```

---

## Task 8: Break monolithic main into pipeline stages

**Files:**
- Modify: `myexplainer_train_v2.py` — extract pipeline stages

- [ ] **Step 1: Read `myexplainer_train_v2.py` fully**

- [ ] **Step 2: Extract `load_dataset(config)`**

- [ ] **Step 3: Extract `load_gnn(config, device)`**

- [ ] **Step 4: Extract `split_by_prediction(gnn, dataset, device)`**

- [ ] **Step 5: Extract `mine_subgraphs(config, splits)`**

- [ ] **Step 6: Extract `build_mapped_dataset(patterns, data)`**

- [ ] **Step 7: Extract `train_explainer(config, model, gnn, train_loader, eval_loader)`**

- [ ] **Step 8: Slim down `main()` to ~15-line orchestrator**

- [ ] **Step 9: Verify full pipeline**

```bash
python myexplainer_train_v2.py --dataset ba2motif --train_mode False --device cpu
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: extract pipeline stages from monolithic main

- 7 composable functions with explicit inputs/outputs
- main() reduced to ~15-line orchestrator
- Each stage independently testable"
```
