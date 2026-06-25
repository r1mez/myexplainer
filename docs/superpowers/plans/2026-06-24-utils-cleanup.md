# Utils 目录整理计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 整理 `utils/` 目录，删除死模块，合并职责重叠的文件，重命名模糊的模块名。

**Tech Stack:** Python 3.9

## Global Constraints

- 所有改动必须保持向后兼容（通过 re-export shim 或更新导入路径）
- 远程服务器上运行，本地无法 import torch — 用 `py_compile` 验证语法
- 不改变任何函数的行为，只做重命名/移动/删除

## 当前状态

```
utils/
├── __init__.py          (30行)  公共 API：get_datasets, set_seed, Gtest, Gtrain, train_collate_fn
├── helper.py            (18行)  set_seed
├── simple_yaml.py      (159行)  YAML 解析器
├── node_labels.py      (207行)  节点标签推断
├── chemistry.py        (307行)  化学转换（atom/bond maps, SMARTS↔PyG）
├── graph_ops.py        (229行)  纯图论操作（extract_explanatory_subgraph）
├── output_conversion.py (71行)  模型输出→PyG Batch
├── dataset_registry.py  (89行)  数据集注册表
├── dataset.py           (35行)  get_datasets 工厂函数
├── subgraph_utils.py    (61行)  to_nx, generate_node_mappings
├── subgraph_method.py  (369行)  subgraph_mining, GraphRepModel, graphsampler
├── pair_data.py        (258行)  MappedDataset, train_collate_fn, nx_to_igraph
├── batch_utils.py      (171行)  output_to_batch, core_data_from_batch
├── baseline_eval_metrics.py (111行) OracleWrappedModel, 单图指标
├── train_utils.py       (61行)  Gtrain, Gtest（GNN 训练循环）
├── train_myexplainer.py (226行)  MyExplainerV2 训练循环
├── vis_utils.py        (279行)  可视化
├── graph_utils.py       (14行)  ⚠️ re-export shim，无活跃消费者
└── tuning.py           (181行)  ⚠️ 死代码，无任何导入
```

## 问题分析

| 问题 | 文件 | 说明 |
|------|------|------|
| 死代码 | `tuning.py` | 181 行，无任何模块导入它 |
| 无用 shim | `graph_utils.py` | 14 行 re-export，无活跃消费者 |
| 名称模糊 | `batch_utils.py` | "batch_utils" 不说明做什么，实际是 output→Batch 转换 |
| 职责重叠 | `subgraph_utils.py` + `pair_data.py` | `to_nx`/`generate_node_mappings` 在 subgraph_utils，但只被 pair_data 使用 |
| 放错位置 | `baseline_eval_metrics.py` | `OracleWrappedModel` 是 eval 基础设施，应归入 `eval/` |
| 名称模糊 | `train_utils.py` | `Gtrain`/`Gtest` 是 GNN 训练工具，名称太泛 |

---

## Task 1: 删除死代码

**Files:**
- Delete: `utils/tuning.py`
- Delete: `utils/graph_utils.py`

**原因:**
- `tuning.py`：无任何活跃模块导入它（`apply_dataset_tuning`, `resolve_gnn_checkpoint`, `summarize_tuning` 均未使用）
- `graph_utils.py`：14 行 re-export shim，搜索确认无活跃消费者

- [ ] **Step 1: 确认无活跃导入**

```bash
grep -rn "from utils.tuning\|import utils.tuning\|from utils.graph_utils\|import utils.graph_utils" --include="*.py" | grep -v __pycache__
```

预期：无输出（或仅匹配文件自身）

- [ ] **Step 2: 删除文件**

```bash
rm utils/tuning.py utils/graph_utils.py
```

- [ ] **Step 3: 验证语法**

```bash
python -m py_compile utils/__init__.py
python -m py_compile config/explainer_config.py
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: delete dead utils/tuning.py and utils/graph_utils.py shim"
```

---

## Task 2: `batch_utils.py` → `output_to_batch.py`

**原因:** `batch_utils` 名称太泛。文件实际做两件事：`output_to_batch`（模型输出→CF Batch）和 `core_data_from_batch`（V1 遗留的子图提取）。重命名为更精确的名称。

**Files:**
- Rename: `utils/batch_utils.py` → `utils/output_to_batch.py`
- Update imports in: `evaluationV2.py`, `case_study/tsne_indistribution_vis.py`

- [ ] **Step 1: 重命名**

```bash
git mv utils/batch_utils.py utils/output_to_batch.py
```

- [ ] **Step 2: 更新导入**

`evaluationV2.py`:
```python
# 改前
from utils.batch_utils import output_to_batch
# 改后
from utils.output_to_batch import output_to_batch
```

`case_study/tsne_indistribution_vis.py`:
```python
# 改前
from utils.batch_utils import output_to_batch
# 改后
from utils.output_to_batch import output_to_batch
```

- [ ] **Step 3: 创建 re-export shim（可选，如果有外部脚本引用）**

检查是否有其他文件导入 `utils.batch_utils`：
```bash
grep -rn "from utils.batch_utils\|import utils.batch_utils" --include="*.py" | grep -v __pycache__
```

- [ ] **Step 4: 验证**

```bash
python -m py_compile utils/output_to_batch.py
python -m py_compile evaluationV2.py
```

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: rename batch_utils.py to output_to_batch.py"
```

---

## Task 3: 合并 `subgraph_utils.py` 到 `pair_data.py`

**原因:** `subgraph_utils.py` 只有 61 行，定义了 `to_nx` 和 `generate_node_mappings`，且只被 `pair_data.py` 使用。它们是 `MappedDataset` 的内部实现细节，不是独立的公共工具。

**Files:**
- Modify: `utils/pair_data.py` — 吸收 `to_nx` 和 `generate_node_mappings`
- Delete: `utils/subgraph_utils.py`

- [ ] **Step 1: 读取两个文件**

```bash
cat utils/subgraph_utils.py
```

将 `to_nx` 和 `generate_node_mappings` 的实现复制到 `pair_data.py` 顶部（在现有导入之后）。

- [ ] **Step 2: 更新 `pair_data.py` 导入**

删除：
```python
from utils.subgraph_utils import generate_node_mappings, to_nx
```

- [ ] **Step 3: 删除 `subgraph_utils.py`**

```bash
rm utils/subgraph_utils.py
```

- [ ] **Step 4: 确认无其他消费者**

```bash
grep -rn "from utils.subgraph_utils\|import utils.subgraph_utils" --include="*.py" | grep -v __pycache__
```

预期：无输出

- [ ] **Step 5: 验证**

```bash
python -m py_compile utils/pair_data.py
```

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "refactor: merge subgraph_utils.py into pair_data.py"
```

---

## Task 4: 移动 `baseline_eval_metrics.py` 到 `eval/`

**原因:** `baseline_eval_metrics.py` 包含 `OracleWrappedModel`（oracle 调用计数包装器）和单图指标函数。这些都是评估基础设施，应归入 `eval/` 包。

**Files:**
- Move: `utils/baseline_eval_metrics.py` → `eval/baseline_eval_metrics.py`
- Update imports in: `models/atex_cf.py`, `models/c2explainer.py`, `models/cf_gnnexplainer.py`

- [ ] **Step 1: 移动文件**

```bash
git mv utils/baseline_eval_metrics.py eval/baseline_eval_metrics.py
```

- [ ] **Step 2: 更新导入**

3 个 baseline 模型文件：
```python
# 改前
from utils.baseline_eval_metrics import OracleWrappedModel
# 改后
from eval.baseline_eval_metrics import OracleWrappedModel
```

- [ ] **Step 3: 更新 `eval/__init__.py`**

添加 re-export：
```python
from eval.baseline_eval_metrics import OracleWrappedModel
```

- [ ] **Step 4: 验证**

```bash
python -m py_compile eval/baseline_eval_metrics.py
python -m py_compile eval/__init__.py
python -m py_compile models/atex_cf.py
```

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: move baseline_eval_metrics.py to eval/ package"
```

---

## Task 5: 重命名 `train_utils.py` → `gnn_train_loop.py`

**原因:** `train_utils` 名称太泛。文件只包含 `Gtrain` 和 `Gtest` — GNN 分类器的通用训练/测试循环。重命名为 `gnn_train_loop` 更精确。

**Files:**
- Rename: `utils/train_utils.py` → `utils/gnn_train_loop.py`
- Update: `utils/__init__.py`

- [ ] **Step 1: 重命名**

```bash
git mv utils/train_utils.py utils/gnn_train_loop.py
```

- [ ] **Step 2: 更新 `utils/__init__.py`**

```python
# 改前
from .train_utils import Gtest, Gtrain
# 改后
from .gnn_train_loop import Gtest, Gtrain
```

- [ ] **Step 3: 验证**

```bash
python -m py_compile utils/gnn_train_loop.py
python -m py_compile utils/__init__.py
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "refactor: rename train_utils.py to gnn_train_loop.py"
```

---

## 最终状态

```
utils/
├── __init__.py          公共 API（不变）
├── helper.py            set_seed（不变）
├── simple_yaml.py       YAML 解析器（不变）
├── node_labels.py       节点标签推断（不变）
├── chemistry.py         化学转换（不变）
├── graph_ops.py         图论操作（不变）
├── output_conversion.py 模型输出转换（不变）
├── dataset_registry.py  数据集注册表（不变）
├── dataset.py           get_datasets（不变）
├── subgraph_method.py   子图挖掘（不变）
├── pair_data.py         MappedDataset + to_nx + generate_node_mappings
├── output_to_batch.py   output_to_batch（原 batch_utils.py）
├── gnn_train_loop.py    Gtrain/Gtest（原 train_utils.py）
├── train_myexplainer.py MyExplainerV2 训练（不变）
├── vis_utils.py         可视化（不变）
│
│  已删除：
├── ~~graph_utils.py~~   re-export shim（无消费者）
├── ~~tuning.py~~        死代码
└── ~~subgraph_utils.py~~ 合并到 pair_data.py
│
│  已移出：
└── ~~baseline_eval_metrics.py~~ → eval/baseline_eval_metrics.py
```
