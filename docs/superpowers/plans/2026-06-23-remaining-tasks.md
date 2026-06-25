# Architecture Refactor — Remaining Tasks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the 7 architecture improvements identified in the code review that were NOT done in the prior refactoring round.

**Architecture:** Each task is self-contained and produces a working codebase. Tasks are ordered by dependency: cleanup first, then extract shared modules, then define interfaces.

**Tech Stack:** Python 3.9, PyTorch 1.10.1, PyTorch Geometric 2.0.4

## Global Constraints

- Binary classification only (`num_classes = 2`)
- Seed fixed to 42 via `set_seed(42)`
- No formal test framework — verify via `python -c "from module import X; print('OK')"`
- GNN checkpoints at `param/gnns/{dataset}_gcn.pt`
- All GNN classes expose: `forward`, `get_pred`, `get_pred_explain`, `get_node_reps`, `get_graph_rep`
- Remote server: conda env `myexplainer`, Python 3.9.21

## Current State (verified)

The following files/features exist and have NOT been modified by prior work:
- `utils/pair_data.py` — `custom_collate_fn` still exists (line 189)
- `utils/loss_hparams.py` — `apply_loss_hparams` still exists (line 86)
- `utils/explainer_hparams.py` — `apply_explainer_hparams` still exists (line 74)
- `evaluationV2.py` — `visualize_explainer_graph` called unconditionally (line 67)
- `models/cf_gnnexplainer.py` — ~155 lines commented-out code (lines 223-377)
- `utils/vis_utils.py` — imports from `graph_editor.metadata` (line 9)
- `gnns/` — all 9 classes inherit from `torch.nn.Module`, no base class
- `models/` — no `__init__.py`, no `base.py`, no base class
- `utils/graph_utils.py` — 599-line monolith, NOT split
- `config/explainer_config.py` — no `grad_clip_max_norm`, no `scheduler_*` fields

---

## Task A: Remove dead code and debugging artifacts

**Files:**
- Modify: `utils/pair_data.py` — delete `custom_collate_fn` (lines 189-204)
- Modify: `utils/loss_hparams.py` — delete `apply_loss_hparams` (lines 86-94)
- Modify: `utils/explainer_hparams.py` — delete `apply_explainer_hparams` (lines 74-83)
- Modify: `evaluationV2.py` — guard `visualize_explainer_graph` behind flag, remove debug print
- Modify: `models/cf_gnnexplainer.py` — delete commented-out block (lines 223-377)

**Interfaces:**
- Consumes: nothing
- Produces: cleaner codebase, no behavioral change

- [ ] **Step 1: Delete `custom_collate_fn` from `utils/pair_data.py`**

Read `utils/pair_data.py`. Delete the `custom_collate_fn` function (lines 189-204). This function handles `ori_graph`/`tgt_graph` format from V1 — it is NOT `train_collate_fn`.

- [ ] **Step 2: Delete `apply_loss_hparams` from `utils/loss_hparams.py`**

Read `utils/loss_hparams.py`. Delete `apply_loss_hparams` (lines 86-94). It mutates an argparse namespace and is never called. Keep `load_loss_hparams`.

- [ ] **Step 3: Delete `apply_explainer_hparams` from `utils/explainer_hparams.py`**

Read `utils/explainer_hparams.py`. Delete `apply_explainer_hparams` (lines 74-83). Keep `load_explainer_hparams`.

- [ ] **Step 4: Guard visualization in `evaluationV2.py`**

Read `evaluationV2.py`. Find the `visualize_explainer_graph` call (line 67). Wrap it:

```python
if getattr(config, 'visualize', False):
    visualize_explainer_graph(...)
```

Also remove the debug print block (lines 72-83, the `if batch_idx == 0:` block with `[DEBUG]` output).

- [ ] **Step 5: Delete commented-out code in `models/cf_gnnexplainer.py`**

Read `models/cf_gnnexplainer.py`. Delete lines 223-377 (the commented-out `evaluate_cf_gnnexplainer` function).

- [ ] **Step 6: Verify**

```bash
python -c "from utils.pair_data import train_collate_fn; print('pair_data OK')"
python -c "from utils.loss_hparams import load_loss_hparams; print('loss_hparams OK')"
python -c "from utils.explainer_hparams import load_explainer_hparams; print('explainer_hparams OK')"
python -c "from evaluationV2 import evaluate; print('evaluationV2 OK')"
python -c "from models.cf_gnnexplainer import CFExplainer; print('cf_gnnexplainer OK')"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: remove dead code and guard debugging artifacts

- Delete unused custom_collate_fn (V1 leftover)
- Delete apply_loss_hparams / apply_explainer_hparams (never called)
- Guard visualize_explainer_graph behind config.visualize flag
- Remove debug print block in evaluationV2.py
- Delete ~155 lines commented-out code in cf_gnnexplainer.py"
```

---

## Task B: Extract `node_labels.py` — fix reverse dependency

**Files:**
- Create: `utils/node_labels.py` — shared node label inference
- Modify: `utils/vis_utils.py` — import from `node_labels` instead of `graph_editor`
- Modify: `graph_editor/metadata.py` — import from `node_labels` for shared logic

**Interfaces:**
- Consumes: `utils/simple_yaml.load_yaml_file`
- Produces:
  - `utils/node_labels.infer_feature_mode(features) -> str`
  - `utils/node_labels.infer_node_label(feature, feature_mode, feature_labels, node_id, dataset) -> str`
  - `utils/node_labels.infer_node_labels_for_dataset(features, dataset, feature_mode, node_ids) -> List[str]`
  - `utils/node_labels.feature_labels_for_dataset(dataset, x_dim, feature_mode) -> List[str]`
  - `utils/node_labels.node_label_mode_for_dataset(dataset, feature_mode, x_dim) -> str`

- [ ] **Step 1: Read `graph_editor/metadata.py` fully**

Understand all functions and their dependencies. The key functions that `vis_utils.py` needs are: `infer_feature_mode`, `infer_node_labels_for_dataset`. The config loading uses `utils/simple_yaml.load_yaml_file` and `configs/node_label_config.yaml`.

- [ ] **Step 2: Create `utils/node_labels.py`**

Extract the node-label inference logic. The functions depend on `configs/node_label_config.yaml` via `load_yaml_file`.

```python
"""Node label inference for graph visualization.

Shared between utils/vis_utils.py and graph_editor/metadata.py
to avoid a reverse dependency from utils -> graph_editor.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.simple_yaml import load_yaml_file

NODE_LABEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "node_label_config.yaml"


@lru_cache(maxsize=1)
def _load_node_label_config() -> Dict[str, Any]:
    payload = load_yaml_file(NODE_LABEL_CONFIG_PATH)
    if not isinstance(payload, dict):
        raise ValueError(f"Node label config must be a YAML mapping: {NODE_LABEL_CONFIG_PATH}")
    return payload


def _dataset_config(dataset: str) -> Dict[str, Any]:
    datasets = _load_node_label_config().get("datasets", {})
    key = str(dataset).strip().lower()
    if key not in datasets:
        raise ValueError(f"Dataset '{dataset}' not found in node label config")
    return datasets[key]


def _feature_is_onehot(features) -> bool:
    """Heuristic: if feature dim <= 20, likely onehot."""
    return features.dim() == 2 and features.size(1) <= 20


def infer_feature_mode(features) -> str:
    """Detect 'onehot' or 'vector' feature mode."""
    return "onehot" if _feature_is_onehot(features) else "vector"


def _atomic_num_to_symbol_map() -> Dict[int, str]:
    """BBBP atomic number -> symbol mapping."""
    return {
        6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I",
    }


def feature_labels_for_dataset(dataset: str, x_dim: int, feature_mode: str) -> List[str]:
    """Return feature label list for a dataset."""
    cfg = _dataset_config(dataset)
    if feature_mode == "onehot":
        labels = cfg.get("labels", [])
        if labels:
            return labels
    return [f"f{i}" for i in range(x_dim)]


def node_label_mode_for_dataset(dataset: str, feature_mode: str, x_dim: int) -> str:
    """Return label rendering mode: 'onehot', 'atomic_num', or 'node_id'."""
    cfg = _dataset_config(dataset)
    override = cfg.get("mode")
    if override:
        return override
    return "onehot" if feature_mode == "onehot" else "node_id"


def infer_node_label(feature, feature_mode: str, feature_labels: List[str], node_id: int = 0, dataset: str = "") -> str:
    """Infer a human-readable label for a single node."""
    mode = node_label_mode_for_dataset(dataset, feature_mode, feature.size(0) if feature.dim() == 1 else feature.size(1))

    if mode == "atomic_num":
        atomic_num = int(feature[0].item()) if feature.dim() == 1 else int(feature.argmax().item())
        symbol_map = _atomic_num_to_symbol_map()
        return symbol_map.get(atomic_num, f"#{atomic_num}")

    if mode == "onehot" and feature_labels:
        idx = int(feature.argmax().item())
        return feature_labels[idx] if idx < len(feature_labels) else f"?{idx}"

    return f"n{node_id}"


def infer_node_labels_for_dataset(features, dataset: str, feature_mode: str, node_ids: Optional[List[int]] = None) -> List[str]:
    """Infer labels for all nodes."""
    if node_ids is None:
        node_ids = list(range(features.size(0)))
    x_dim = features.size(1)
    labels = feature_labels_for_dataset(dataset, x_dim, feature_mode)
    return [infer_node_label(features[i], feature_mode, labels, node_ids[i], dataset) for i in range(len(node_ids))]
```

- [ ] **Step 3: Update `utils/vis_utils.py`**

Change line 9 from:
```python
from graph_editor.metadata import infer_feature_mode, infer_node_labels_for_dataset
```
to:
```python
from utils.node_labels import infer_feature_mode, infer_node_labels_for_dataset
```

- [ ] **Step 4: Update `graph_editor/metadata.py`**

Replace the implementations of `infer_feature_mode`, `infer_node_label`, `infer_node_labels_for_dataset`, `feature_labels_for_dataset`, `node_label_mode_for_dataset` with imports from `utils.node_labels`. Keep the re-exports so `graph_editor` callers don't break. Keep `resolve_model_path`, `normalize_dataset_name`, `is_supported_dataset`, `is_supported_split`, `SUPPORTED_DATASETS`, `SUPPORTED_SPLITS`, and the `atom_type_options_for_dataset`, `default_atom_type_for_dataset`, `bbbp_atom_symbol_from_feature` functions in `metadata.py` — they are editor-specific.

- [ ] **Step 5: Verify no circular imports**

```bash
python -c "from utils.node_labels import infer_feature_mode; print('node_labels OK')"
python -c "from graph_editor.metadata import infer_feature_mode; print('metadata OK')"
python -c "from utils.vis_utils import visualize_explainer_graph; print('vis_utils OK')"
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

## Task C: Extract GNN base class

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
        - _forward_convs(x, edge_index, edge_weight): node embeddings [N, hidden_dim]
        - _pool(node_emb, batch): graph embeddings [B, hidden_dim]
        - _classify(graph_emb): logits [B, num_classes]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    @abstractmethod
    def _forward_convs(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        """Run convolution layers. Returns node embeddings [N, hidden_dim]."""
        ...

    @abstractmethod
    def _pool(self, node_emb: Tensor, batch: Tensor) -> Tensor:
        """Pool node embeddings to graph embedding [B, hidden_dim]."""
        ...

    @abstractmethod
    def _classify(self, graph_emb: Tensor) -> Tensor:
        """Classifier head. Returns logits [B, num_classes]."""
        ...

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Forward pass: encode -> pool -> classify. Returns logits."""
        node_emb = self._forward_convs(x, edge_index)
        graph_emb = self._pool(node_emb, batch)
        return self._classify(graph_emb)

    def get_node_reps(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        """Get node-level representations."""
        return self._forward_convs(x, edge_index, edge_weight)

    def get_graph_rep(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_weight: Tensor = None) -> Tensor:
        """Get graph-level representation (before classifier head)."""
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        return self._pool(node_emb, batch)

    def get_pred(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (softmax_probs, logits)."""
        logits = self.forward(x, edge_index, batch)
        return F.softmax(logits, dim=-1), logits

    def get_pred_explain(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (softmax_probs, logits) with edge weights for explanation."""
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        graph_emb = self._pool(node_emb, batch)
        logits = self._classify(graph_emb)
        return F.softmax(logits, dim=-1), logits
```

- [ ] **Step 2: Migrate `gnns/ba2motif_gnn.py`**

Read the file. Current structure:
- `get_node_reps(x, edge_index, edge_weight=None)` — runs conv+bn+act+dropout loop
- `get_graph_rep(x, edge_index, batch, edge_weight=None)` — calls `get_node_reps` then `global_mean_pool`
- `classifier(graph_x)` — `lin1 -> relu -> lin2`
- `forward(x, edge_index, batch)` — `get_graph_rep` then `classifier`
- `get_pred(x, edge_index, batch)` — `forward` then softmax
- `get_pred_explain(x, edge_index, edge_weight, batch)` — `get_graph_rep` with `edge_weight` then `classifier`

Change `class BA2MotifGCN(torch.nn.Module)` to `class BA2MotifGCN(BaseGNNClassifier)`. Add `from gnns.base import BaseGNNClassifier`. Implement:
- `_forward_convs` = current `get_node_reps` body
- `_pool` = `global_mean_pool(node_emb, batch)`
- `_classify` = current `classifier` body

Delete the 5 duplicated methods (`forward`, `get_pred`, `get_pred_explain`, `get_node_reps`, `get_graph_rep`). Keep `classifier` if other code calls it directly, or fold it into `_classify`.

- [ ] **Step 3: Migrate `gnns/mutag_gnn.py`**

Same pattern. Note: `in_channels` hardcoded to 14. `get_node_reps`/`get_graph_rep` don't accept `edge_weight` — the base class handles this uniformly.

- [ ] **Step 4: Migrate `gnns/bbbp_gnn.py`**

Same pattern. Keep extra `get_emb` method if it exists. `in_channels` hardcoded to 9.

- [ ] **Step 5: Migrate `gnns/nci1_gnn.py`**

Same pattern. Uses `LEConv` instead of `GCNConv`. `in_channels` hardcoded to 37.

- [ ] **Step 6: Migrate `gnns/benzene_gcn.py`**

Same pattern. `in_channels` hardcoded to 14.

- [ ] **Step 7: Migrate `gnns/proteins_gnn.py`**

Same pattern. Uses `global_max_pool` instead of `global_mean_pool`.

- [ ] **Step 8: Migrate `gnns/mutag188_gnn.py`**

Same pattern.

- [ ] **Step 9: Migrate `gnns/alkane_carbonyl_gnn.py`**

Same pattern. Uses `EdgeWeightedGATConv`, residual connections, configurable pooling (`"mean"` or `"mean_max"`).

- [ ] **Step 10: Migrate `gnns/fluoride_carbonyl_gnn.py`**

Structurally identical to alkane_carbonyl.

- [ ] **Step 11: Update `gnns/__init__.py`**

Add `from gnns.base import BaseGNNClassifier` to exports.

- [ ] **Step 12: Verify**

```bash
python -c "
from gnns import BaseGNNClassifier
from gnns import BA2MotifGCN, Mutag_GCN, BBBP_GCN, NCI1GCN
from gnns import Benzene_GCN, PROTEINSGCN, Mutag188_GCN
from gnns import AlkaneCarbonylGCN, FluorideCarbonylGCN
for cls in [BA2MotifGCN, Mutag_GCN, BBBP_GCN, NCI1GCN, Benzene_GCN, PROTEINSGCN, Mutag188_GCN, AlkaneCarbonylGCN, FluorideCarbonylGCN]:
    assert issubclass(cls, BaseGNNClassifier), f'{cls.__name__} does not extend BaseGNNClassifier'
print('All 9 GNN classes extend BaseGNNClassifier')
"
```

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: extract BaseGNNClassifier base class

- All 9 GNN classes now extend BaseGNNClassifier
- 5 interface methods implemented once in base class
- Subclasses implement 3 abstract methods: _forward_convs, _pool, _classify
- Eliminates ~450 lines of duplicated method code"
```

---

## Task D: Split `utils/graph_utils.py` into focused modules

**Files:**
- Create: `utils/chemistry.py` — atom/bond maps, `smarts_to_data`, `data_to_mol`
- Create: `utils/graph_ops.py` — `extract_explanatory_subgraph`, `exclude_explanatory_subgraph`
- Create: `utils/output_conversion.py` — `process_outputs`
- Modify: `utils/graph_utils.py` — keep as re-export layer for backward compat
- Modify: `eval/metrics.py` — import from `graph_ops`
- Modify: `evaluationV2.py` — import from `graph_ops`
- Modify: `utils/train_utils.py` — import from `graph_ops`
- Modify: `models/myexplainerV2.py` — import from `output_conversion` (if it imports `process_outputs`)

**Interfaces:**
- `utils/chemistry.py`:
  - `MUTAG_atom_map`, `BBBP_atom_map`, `MUTAG188_atom_map`, `atom_map`, `bond_map`, `atom_idx_map`
  - `smarts_to_data(dataset_name, smarts) -> Data`
  - `data_to_mol(dataset_name, data) -> Mol`
  - `smarts_to_mol(smarts) -> Mol`
- `utils/graph_ops.py`:
  - `extract_explanatory_subgraph(original, counterfactual) -> Data|Batch|List[Data]`
  - `exclude_explanatory_subgraph(original, counterfactual) -> Data|Batch|List[Data]`
- `utils/output_conversion.py`:
  - `process_outputs(args, outputs) -> Batch`

- [ ] **Step 1: Create `utils/chemistry.py`**

Read `utils/graph_utils.py`. Extract lines 1-280 (atom/bond maps, `smarts_to_data`, `data_to_mol`, `smarts_to_mol`, `_correct_valence`, `_sanitize_with_valence_correction`) into `utils/chemistry.py`. Keep all imports needed by these functions (rdkit, networkx, torch_geometric).

- [ ] **Step 2: Create `utils/graph_ops.py`**

Extract `extract_explanatory_subgraph` and `exclude_explanatory_subgraph` from `graph_utils.py`. These are pure graph-theoretic operations that depend only on `torch` and `torch_geometric.data`. Extract the shared `_normalize_input` helper (it's duplicated inside both functions — deduplicate).

- [ ] **Step 3: Create `utils/output_conversion.py`**

Extract `process_outputs` from `graph_utils.py`.

- [ ] **Step 4: Update `utils/graph_utils.py` as re-export layer**

Replace the body with re-exports:

```python
"""Backward-compatible re-exports. Prefer importing from chemistry, graph_ops, output_conversion."""
from utils.chemistry import *  # noqa: F401,F403
from utils.graph_ops import extract_explanatory_subgraph, exclude_explanatory_subgraph  # noqa: F401
from utils.output_conversion import process_outputs  # noqa: F401
```

- [ ] **Step 5: Update internal imports**

- `eval/metrics.py`: change `from utils.graph_utils import extract_explanatory_subgraph` to `from utils.graph_ops import extract_explanatory_subgraph`
- `evaluationV2.py`: change `from utils.graph_utils import extract_explanatory_subgraph` to `from utils.graph_ops import extract_explanatory_subgraph`
- `utils/train_utils.py`: change `from utils.graph_utils import ...` to `from utils.graph_ops import ...`
- Check if `models/myexplainerV2.py` imports `process_outputs` — if so, change to `from utils.output_conversion import process_outputs`

- [ ] **Step 6: Verify**

```bash
python -c "from utils.chemistry import smarts_to_data, data_to_mol; print('chemistry OK')"
python -c "from utils.graph_ops import extract_explanatory_subgraph; print('graph_ops OK')"
python -c "from utils.output_conversion import process_outputs; print('output_conversion OK')"
python -c "from utils.graph_utils import extract_explanatory_subgraph; print('backward compat OK')"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: split graph_utils.py into focused modules

- utils/chemistry.py: atom/bond maps, RDKit conversions
- utils/graph_ops.py: extract_explanatory_subgraph, exclude (pure graph ops)
- utils/output_conversion.py: process_outputs
- graph_utils.py kept as re-export layer for backward compat"
```

---

## Task E: Define BaseExplainer interface and unify evaluation

**Files:**
- Create: `models/__init__.py`
- Create: `models/base.py` — `BaseExplainer`, `CFResult`
- Modify: `models/atex_cf.py` — extend `BaseExplainer`, delete `evaluate_atex_cf_graph`
- Modify: `models/c2explainer.py` — extend `BaseExplainer`, delete `evaluate_c2_structural`
- Modify: `models/cf_gnnexplainer.py` — extend `BaseExplainer`, delete `evaluate_cf_gnnexplainer`
- Modify: `models/clear.py` — extend `BaseExplainer`, delete `evaluate_graphcfe`
- Modify: `models/rsgg_ce.py` — extend `BaseExplainer`, delete `evaluate_rsgg_ce`
- Modify: `eval/baseline_runner.py` — update to use `BaseExplainer.explain_graph()` + `CFResult`

**Interfaces:**
- `models/base.py`:
  ```python
  @dataclass
  class CFResult:
      cf_edge_index: Tensor       # [2, E_cf]
      cf_edge_weight: Tensor      # [E_cf]
      oracle_calls: int = 0
      runtime: float = 0.0

  class BaseExplainer(nn.Module):
      def explain_graph(self, data: Data, device: str = "cpu") -> CFResult: ...
      def fit(self, train_dataset, gnn, device: str = "cpu") -> None: ...
  ```

- [ ] **Step 1: Create `models/__init__.py`**

```python
"""Counterfactual graph explainer models."""
```

- [ ] **Step 2: Create `models/base.py`**

```python
"""Base interface for counterfactual graph explainers."""
from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class CFResult:
    """Result of a counterfactual explanation."""
    cf_edge_index: Tensor       # [2, E_cf]
    cf_edge_weight: Tensor      # [E_cf]
    oracle_calls: int = 0
    runtime: float = 0.0


class BaseExplainer(nn.Module):
    """Abstract base for counterfactual explainers.

    Subclasses must implement:
        - explain_graph(data, device) -> CFResult
    Subclasses may optionally implement:
        - fit(train_dataset, gnn, device) for trainable explainers
    """

    def explain_graph(self, data: Data, device: str = "cpu") -> CFResult:
        raise NotImplementedError

    def fit(self, train_dataset, gnn, device: str = "cpu") -> None:
        pass
```

- [ ] **Step 3: Migrate `models/atex_cf.py`**

Add `from models.base import BaseExplainer, CFResult`. Change `class ATEXCFExplainer(nn.Module)` to `class ATEXCFExplainer(BaseExplainer)`. Implement `explain_graph` that returns `CFResult`. Delete `evaluate_atex_cf_graph` function. Update `__main__` block.

- [ ] **Step 4: Migrate `models/c2explainer.py`**

Same pattern. Delete `evaluate_c2_structural`. Delete duplicate `visualize_comparison` if present.

- [ ] **Step 5: Migrate `models/cf_gnnexplainer.py`**

Same pattern. `run_one_graph` returns dense adj — `explain_graph` converts to sparse `edge_index` via `dense_to_sparse`.

- [ ] **Step 6: Migrate `models/clear.py`**

Same pattern. `fit()` calls the existing training functions. `explain_graph` wraps the existing CF generation logic.

- [ ] **Step 7: Migrate `models/rsgg_ce.py`**

Same pattern. `fit()` calls `train_rsgg_ce()`.

- [ ] **Step 8: Update `eval/baseline_runner.py`**

Update `BaselineRunner.run()` to accept a `BaseExplainer`, call `explainer.explain_graph(data, device)`, and build CF graph from `CFResult`.

- [ ] **Step 9: Verify**

```bash
python -c "
from models.base import BaseExplainer, CFResult
from models.atex_cf import ATEXCFExplainer
from models.c2explainer import C2ExplainerStructuralOnly
from models.cf_gnnexplainer import CFExplainer
from models.clear import GraphCFE
from models.rsgg_ce import RSGGCEExplainer
for cls in [ATEXCFExplainer, C2ExplainerStructuralOnly, CFExplainer, GraphCFE, RSGGCEExplainer]:
    assert issubclass(cls, BaseExplainer), f'{cls.__name__} does not extend BaseExplainer'
print('All 5 baseline models extend BaseExplainer')
"
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: define BaseExplainer interface and unify evaluation

- models/base.py: BaseExplainer with explain_graph() -> CFResult
- All 5 baseline models extend BaseExplainer
- CFResult dataclass: cf_edge_index, cf_edge_weight, oracle_calls, runtime
- Deleted ~580 lines of duplicated evaluation functions
- BaselineRunner updated to use BaseExplainer interface"
```

---

## Task F: Consolidate configuration

**Files:**
- Modify: `config/explainer_config.py` — add `grad_clip_max_norm`, `scheduler_*` fields
- Modify: `myexplainer_train_v2.py` — read scheduler params from config
- Modify: `utils/train_myexplainer.py` — read `grad_clip_max_norm` from config
- Delete: `utils/loss_hparams.py` — merged into config
- Delete: `utils/explainer_hparams.py` — merged into config

**Interfaces:**
- `ExplainerConfig` gains:
  - `grad_clip_max_norm: float = 1.0`
  - `scheduler_factor: float = 0.8`
  - `scheduler_patience: int = 15`
  - `scheduler_min_lr: float = 1e-6`

- [ ] **Step 1: Add new fields to `ExplainerConfig`**

Read `config/explainer_config.py`. Add the new fields with defaults matching current hardcoded values.

- [ ] **Step 2: Inline YAML loading into `ExplainerConfig.from_args`**

The `from_args` method currently calls `load_loss_hparams` and `load_explainer_hparams`. Inline that logic (it's just `load_yaml_file` + key extraction) so the separate modules are no longer needed.

- [ ] **Step 3: Update `myexplainer_train_v2.py`**

Replace hardcoded scheduler params (`factor=0.8, patience=15, min_lr=1e-6`) with `config.scheduler_*`.

- [ ] **Step 4: Update `utils/train_myexplainer.py`**

Replace hardcoded `max_norm=1.0` with `config.grad_clip_max_norm`.

- [ ] **Step 5: Delete `utils/loss_hparams.py` and `utils/explainer_hparams.py`**

Remove both files. Update any imports that reference them.

- [ ] **Step 6: Verify**

```bash
python -c "
from config import ExplainerConfig
import argparse
args = argparse.Namespace(dataset='ba2motif', cuda='0', device='cpu', epochs=1, lr=0.01, h_dim=256, z_dim=32, batch_size=256, train_mode='True', task='graph', top_k=1, threshold=0, max_num_nodes=25, dropout=0.1, weight_decay=1e-5, subgraph_method='genGraphEx', gnn_path='param/')
config = ExplainerConfig.from_args(args)
print(f'grad_clip: {config.grad_clip_max_norm}')
print(f'scheduler_factor: {config.scheduler_factor}')
print('Config consolidation OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: consolidate configuration into ExplainerConfig

- Added grad_clip_max_norm, scheduler_factor, scheduler_patience, scheduler_min_lr
- Inlined YAML loading from loss_hparams.py and explainer_hparams.py
- Deleted loss_hparams.py and explainer_hparams.py
- All training params now in one place"
```

---

## Task G: Clean up `utils/__init__.py`

**Files:**
- Modify: `utils/__init__.py` — remove stale exports

**Interfaces:**
- No new interfaces

- [ ] **Step 1: Read `utils/__init__.py`**

Check what it exports. Remove `custom_collate_fn` if present (deleted in Task A). Remove any other stale exports.

- [ ] **Step 2: Verify**

```bash
python -c "from utils import get_datasets, set_seed, train_collate_fn; print('utils OK')"
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: clean up utils/__init__.py exports"
```
