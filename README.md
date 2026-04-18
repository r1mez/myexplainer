# CCFGExplainer

> Class-Conditioned Frequent-Subgraph Guided Counterfactual Explainer for Graph Neural Networks

CCFGExplainer 是一个面向图分类任务的 GNN 反事实解释项目。项目以预训练图神经网络分类器为被解释对象，通过类别条件频繁子图挖掘、候选子图匹配、边删除/边添加生成以及原型对齐约束，生成能够改变原 GNN 预测结果的反事实图，并用 validity、proximity、fidelity、sparsity 等指标评估解释质量。

本项目主要适用于分子图、合成图和常见图分类数据集上的 GNN 可解释性研究。

## 核心思想

给定一个已经训练好的 GNN 分类器和一张输入图，CCFGExplainer 的目标是学习一个尽量小的图结构修改，使修改后的反事实图能够被原 GNN 判为目标类别。当前实现默认二分类场景，目标类别为原预测类别的相反类别。

整体流程如下：

```text
Graph Dataset
    |
    v
Pre-trained GNN Classifier
    |
    v
Class-wise Prediction Split
    |
    v
Frequent Subgraph Mining / Prototype Construction
    |
    v
Subgraph Matching with VF2
    |
    v
CCFGExplainer
  - GCN node encoder
  - DeleteNet for edge keeping/deletion
  - AddVGAE for candidate edge addition
  - Prototype alignment loss
    |
    v
Counterfactual Graphs
    |
    v
Evaluation Metrics
```

## 项目特性

- 面向 GNN 图分类任务的反事实解释生成。
- 支持基于类别预测结果的频繁子图挖掘，为不同类别构造候选模式。
- 使用 VF2 子图同构匹配为每张图定位可解释的频繁子图区域。
- 使用 `DeleteNet` 预测原始边的保留概率，支持学习式边删除。
- 使用 VGAE 风格模块在频繁子图区域内生成候选新增边。
- 引入类别原型对齐约束，使生成后的反事实子图更接近目标类别的结构表示。
- 提供 validity、proximity、fidelity、sparsity、runtime、oracle calls 以及按类别 flip success 的评估输出。
- 内置多种图数据集和预训练 GNN 结构的代码实现。

## 目录结构

```text
.
├── myexplainer_train_v2.py       # 当前训练与评估入口脚本
├── evaluationV2.py               # 反事实图评估逻辑与指标计算
├── models/
│   ├── myexplainerV2.py          # CCFGExplainer 核心模型实现
│   ├── atex_cf.py                # baseline / 对比方法相关实现
│   ├── c2explainer.py            # baseline / 对比方法相关实现
│   ├── cf_gnnexplainer.py        # baseline / 对比方法相关实现
│   ├── clear.py                  # baseline / 对比方法相关实现
│   ├── rsgg_ce.py                # baseline / GRETEL-RSGG-CE 适配实现
│   └── graph_conv.py             # 图卷积相关组件
├── gnns/                         # 被解释的 GNN 分类器结构
├── datasets/                     # 各数据集的 PyTorch Geometric 封装
├── utils/
│   ├── dataset.py                # 数据集加载入口 get_datasets
│   ├── pair_data.py              # 子图匹配数据集与 collate 函数
│   ├── subgraph_method.py        # 频繁子图 / 类别模式生成
│   ├── train_myexplainer.py      # 训练循环、checkpoint 保存、loss 记录
│   ├── graph_utils.py            # 图处理、反事实图转换等工具
│   ├── batch_utils.py            # batch 与输出转换工具
│   └── vis_utils.py              # 可视化工具
├── data/                         # 数据集目录，GitHub 仓库中通常不会包含
├── param/
│   ├── gnns/                     # 预训练 GNN 权重目录
│   └── *.pt                      # CCFGExplainer 训练 checkpoint
└── tmp_files/                    # 临时文件目录
```

说明：部分文件名仍保留早期实验命名，但 README 和项目展示统一使用 `CCFGExplainer` 作为项目名。

## 支持的数据集

当前 `utils.dataset.get_datasets()` 已实现以下数据集入口：

| 参数名 | 数据目录 | 说明 |
| --- | --- | --- |
| `ba2motif` | `data/ba2motif` | BA-2Motif 合成图数据集 |
| `mutag` | `data/mutag` | Mutagenicity / MUTAG 分子图数据集 |
| `nci1` | `data/NCI1` | NCI1 分子图数据集 |
| `bbbp` | `data/bbbp` | BBBP 分子图数据集 |
| `benzene` | `data/benzene` | Benzene 相关分子图数据集 |
| `alkane_carbonyl` | `data/alkane_carbonyl` | Alkane-Carbonyl 分子图数据集 |
| `fluoride_carbonyl` | `data/fluoride_carbonyl` | Fluoride-Carbonyl 分子图数据集 |
| `proteins` | `data/proteins` | PROTEINS 图分类数据集 |

运行前需要确保对应数据集已经处理为 PyTorch Geometric 可读取的格式，例如 `processed/training.pt`、`processed/testing.pt`、`processed/evaluation.pt`。

## 环境依赖

本项目没有固定的 `requirements.txt`，建议使用独立 Conda / venv 环境安装依赖。核心依赖包括：

```text
python >= 3.9
pytorch
torch-geometric
torch-scatter
numpy
scipy
scikit-learn
networkx
python-igraph
matplotlib
tqdm
pandas
rdkit
```

如果使用 CUDA，请安装与本机 CUDA、PyTorch 版本匹配的 `torch-geometric`、`torch-scatter` 等包。PyG 生态包对版本比较敏感，推荐参考 PyTorch Geometric 官方安装方式选择对应 wheel。

示例安装流程：

```bash
conda create -n ccfgexplainer python=3.10
conda activate ccfgexplainer

# 根据本机 CUDA / CPU 环境安装 PyTorch
pip install torch torchvision torchaudio

# 根据 PyTorch 和 CUDA 版本安装 PyG 相关包
pip install torch-geometric torch-scatter

# 安装通用科研依赖
pip install numpy scipy scikit-learn networkx python-igraph matplotlib tqdm pandas rdkit
```

## 数据与权重准备

训练 CCFGExplainer 前，需要准备两类资源。

1. 数据集文件

将数据放置到 `data/` 下对应目录，例如：

```text
data/
├── ba2motif/
├── mutag/
├── NCI1/
├── bbbp/
├── benzene/
├── alkane_carbonyl/
├── fluoride_carbonyl/
└── proteins/
```

2. 预训练 GNN 分类器

训练脚本会从以下路径加载被解释的预训练 GNN：

```text
param/gnns/<dataset>_gcn.pt
```

例如：

```text
param/gnns/ba2motif_gcn.pt
param/gnns/mutag_gcn.pt
param/gnns/nci1_gcn.pt
param/gnns/proteins_gcn.pt
```

注意：当前代码中会使用 `args.dataset.lower()` 拼接权重文件名。在 Linux / macOS 等大小写敏感系统上，请确保权重文件名与小写数据集名一致，例如将 `NCI1_gcn.pt` 命名为 `nci1_gcn.pt`。

## 快速开始

默认配置运行 BA-2Motif：

```bash
python myexplainer_train_v2.py --dataset ba2motif --cuda 0 --epochs 100
```

在 CPU 上运行：

```bash
python myexplainer_train_v2.py --dataset ba2motif --device cpu --epochs 100
```

运行 MUTAG：

```bash
python myexplainer_train_v2.py --dataset mutag --cuda 0 --epochs 100 --batch_size 256
```

运行 PROTEINS：

```bash
python myexplainer_train_v2.py --dataset proteins --cuda 0 --epochs 100 --batch_size 128
```

仅加载已训练的 CCFGExplainer checkpoint 并进行评估：

```bash
python myexplainer_train_v2.py --dataset ba2motif --train_mode False
```

重要说明：`--train_mode` 当前在代码中使用 `type=bool` 解析。命令行传入布尔值时，Python `argparse` 的行为可能不符合直觉。如果需要稳定地切换训练/评估模式，建议后续将该参数改为 `store_true` / `store_false` 或显式字符串解析。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `ba2motif` | 数据集名称 |
| `--cuda` | `0` | GPU 设备编号 |
| `--device` | `cuda` | 设备类型，代码会优先根据 CUDA 可用性设置 |
| `--train_mode` | `True` | 是否训练模型，否则加载已有 checkpoint |
| `--task` | `graph` | 任务类型，当前主流程面向图分类 |
| `--batch_size` | `256` | CCFGExplainer 训练 batch size |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `0.01` | 学习率 |
| `--weight_decay` | `1e-5` | Adam 权重衰减 |
| `--h_dim` | `256` | 图编码器隐藏维度 |
| `--z_dim` | `32` | VGAE 潜变量维度 |
| `--max_num_nodes` | `25` | 最大节点数配置，部分数据集需要按图规模调整 |
| `--dropout` | `0.1` | dropout 比例 |
| `--top_k` | `1` | 相似图配对相关参数，当前主流程中保留 |
| `--threshold` | `0` | 预测置信度阈值相关参数 |
| `--subgraph_sample_threshold` | `None` | 子图采样边概率阈值；未设置时按数据集使用默认值 |
| `--subgraph_method` | `genGraphEx` | 频繁子图生成方法 |
| `--proto_topk` | `100` | 每个类别用于构造 prototype 的 top-k pattern 数量 |
| `--proto_refresh_every` | `5` | 每隔多少个 epoch 刷新一次类别 prototype |
| `--w_proto` | `1.0` | prototype alignment loss 权重 |

其中 `--threshold` 用于预测置信度筛选，`--subgraph_sample_threshold` 用于 `subgraph_method.py` 中的子图采样边概率阈值。若未显式传入 `--subgraph_sample_threshold`，当前默认按数据集选择：`ba2motif=0.97`，`mutag/proteins/alkane_carbonyl/fluoride_carbonyl/nci1=0.70`。

模型内部还支持若干 loss 权重的默认值，例如 `w_cf`、`w_l1_add`、`w_l1_del`、`w_vgae_recon`、`w_vgae_kl`。这些权重目前通过 `getattr(args, ..., default)` 读取，若需要命令行调参，可以在 `parse_args()` 中补充对应参数。

## 训练流程

主入口脚本的实际流程如下：

1. 读取数据集的 train / validation / test split。
2. 加载 `param/gnns/<dataset>_gcn.pt` 中的预训练 GNN，并冻结其参数。
3. 使用预训练 GNN 对训练集做预测，根据预测类别划分样本。
4. 对每个类别分别进行频繁子图生成，得到 class-wise pattern set。
5. 将 pattern 转换为 prototype bank，用于原型对齐约束。
6. 通过 `MappedDataset` 为 train / validation / test 图匹配频繁子图区域。
7. 初始化 CCFGExplainer，并训练以下模块：

```text
GCN Encoder -> DeleteNet + AddVGAE -> Counterfactual Edge Set -> Frozen GNN Prediction
```

8. 在验证集上计算总损失，并保存验证损失最优的 checkpoint。
9. 加载最优 checkpoint，在验证集上输出反事实解释评估指标。

## 输出文件

训练过程中会将 checkpoint 保存到 `param/`：

```text
param/myexplainer_<dataset>_best.pt
param/myexplainer_<dataset>_epoch_<epoch>.pt
```

其中 `best.pt` 是验证损失最优的模型权重，`epoch_<epoch>.pt` 是每 10 个 epoch 保存一次的训练 checkpoint。

说明：checkpoint 文件名来自当前代码实现，文件名前缀不代表项目命名。项目统一命名为 `CCFGExplainer`。

## 评估指标

`evaluationV2.evaluate()` 会返回并打印以下指标：

| 指标 | 含义 |
| --- | --- |
| `validity` | 生成的反事实图成功翻转到目标类别的比例 |
| `successful` / `total` | 成功翻转数量与总样本数量 |
| `proximity` | 原图与反事实图之间的结构距离，当前实现基于邻接矩阵 L1 差异归一化 |
| `fidelity` | 原预测类别概率在反事实图上的下降程度 |
| `sparsity` | 反事实修改的稀疏程度 |
| `runtime` | 平均每张图生成反事实图的耗时 |
| `oracle_calls` | 平均每张图调用被解释 GNN 的次数 |
| `per_class_flip` | 按原始类别统计的 flip success ratio |

示例输出格式：

```text
Evaluation Results on Validation Set:
  Validity -> 0.8123 (successful: 812/total: 1000)
  Proximity -> 0.1432
  Fidelity_prob -> 0.4567
  Sparsity -> 0.7210
  Class 0 Flip Success -> 0.8000 (successful: 400/total: 500, target: 1)
  Class 1 Flip Success -> 0.8240 (successful: 412/total: 500, target: 0)
```

## 可视化

训练和评估过程中会调用 `utils.vis_utils.visualize_explainer_graph()`，用于展示原图、目标标签以及生成后的反事实结构。若在服务器或无 GUI 环境运行，建议将 Matplotlib 后端切换为非交互模式，或关闭可视化调用以避免阻塞训练。

## 代码入口说明

- `myexplainer_train_v2.py`：主训练和评估入口。
- `models/myexplainerV2.py`：CCFGExplainer 的核心模型实现，包括图编码、边删除、候选边添加和 prototype loss。
- `models/rsgg_ce.py`：基于 GRETEL 官方实现思路改写的 RSGG-CE baseline，提供训练、加载和评估接口。
- `utils/subgraph_method.py`：基于类别数据生成频繁子图 pattern。
- `utils/pair_data.py`：将 pattern 匹配回每张图，构造训练需要的 graph-subgraph batch。
- `utils/train_myexplainer.py`：训练循环、验证损失、checkpoint 保存。
- `evaluationV2.py`：反事实图生成后的评估逻辑。

## 注意事项

- 当前主流程默认二分类，代码中 `args.num_classes = 2`，目标标签为 `1 - original_prediction`。
- `param/`、`data/`、`*.pt` 通常体积较大，已经在 `.gitignore` 中忽略；如果在 GitHub 上复现实验，需要单独准备数据集和预训练权重。
- 如果数据集规模明显大于 BA-2Motif，需要根据图大小调整 `--max_num_nodes`、`--batch_size` 和频繁子图采样数量。
- `subgraph_method.py` 中不同数据集的 pattern 采样数量不同，例如 NCI1 默认采样更多 pattern，运行时间会更长。
- 当前本地环境如果缺少 `numpy`、`torch` 或 PyG 相关包，直接运行脚本会失败；请先完成依赖安装。
- 若在 Linux / macOS 上运行，请注意数据目录和权重文件名大小写必须与代码拼接路径一致。

## 适用场景

CCFGExplainer 适合用于以下研究和实验：

- 图神经网络反事实解释。
- 分子图分类模型可解释性分析。
- 频繁子图模式与 GNN 决策边界之间的关系研究。
- 不同 GNN explainer 方法的对比实验。
- 反事实图生成的结构约束、稀疏性与有效性评估。

## 引用

如果你在论文或项目中使用 CCFGExplainer，可以在后续补充正式论文引用信息，例如：

```bibtex
@misc{ccfgexplainer,
  title  = {CCFGExplainer: Class-Conditioned Frequent-Subgraph Guided Counterfactual Explainer for Graph Neural Networks},
  author = {Your Name},
  year   = {2026},
  note   = {GitHub repository}
}
```
