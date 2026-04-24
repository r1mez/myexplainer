import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

from evaluationV2 import evaluate
from models.myexplainerV2 import MyExplainerV2
from utils import get_datasets, train_collate_fn
from utils.loss_hparams import LOSS_HPARAM_KEYS, apply_loss_hparams
from utils.pair_data import MappedDataset
from utils.subgraph_method import subgraph_mining
from utils.train_myexplainer import train_myexplainerV2

from gnns import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)


def pattern_nx_to_data(pattern_nx, x_dim):
    if pattern_nx is None or pattern_nx.number_of_nodes() == 0:
        return None

    node_list = list(pattern_nx.nodes())
    num_nodes = len(node_list)
    node_mapping = {node_id: idx for idx, node_id in enumerate(node_list)}

    x_rows = []
    for node_id in node_list:
        node_attrs = pattern_nx.nodes[node_id]
        feat = node_attrs.get("x")
        if feat is not None:
            feat = torch.as_tensor(feat, dtype=torch.float).view(-1)
        else:
            feat_idx = node_attrs.get("feature", node_attrs.get("type"))
            if feat_idx is None:
                return None
            feat = torch.zeros(x_dim, dtype=torch.float)
            feat_idx = int(feat_idx)
            if feat_idx < 0 or feat_idx >= x_dim:
                return None
            feat[feat_idx] = 1.0

        if feat.numel() != x_dim:
            return None
        x_rows.append(feat)

    if not x_rows:
        return None

    x = torch.stack(x_rows, dim=0)

    if x.size(-1) != x_dim:
        return None

    edges = []
    for src, dst in pattern_nx.edges():
        src_idx = node_mapping[src]
        dst_idx = node_mapping[dst]
        edges.append((src_idx, dst_idx))
        edges.append((dst_idx, src_idx))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def pattern_family_to_nx(pattern_item):
    if isinstance(pattern_item, dict):
        return pattern_item.get("representative_nx")
    return pattern_item


def pattern_family_support(pattern_item):
    if isinstance(pattern_item, dict):
        return float(pattern_item.get("in_class_support", 1.0))
    return 1.0


def pattern_family_score(pattern_item):
    if isinstance(pattern_item, dict):
        return float(pattern_item.get("score", 1.0))
    return 1.0


def pattern_family_weight(pattern_item):
    support = pattern_family_support(pattern_item)
    score = max(pattern_family_score(pattern_item), 0.0)
    return support * score


def build_prototype_bank(patterns, x_dim, topk):
    prototype_bank = {}
    prototype_counts = {}

    if isinstance(patterns, dict):
        pattern_items = patterns.items()
    else:
        pattern_items = enumerate(patterns)

    for class_idx, class_patterns in pattern_items:
        valid_graphs = []
        weights = []
        fallback_weights = []
        scores = []

        for pattern_item in class_patterns[:topk]:
            pattern_nx = pattern_family_to_nx(pattern_item)
            data = pattern_nx_to_data(pattern_nx, x_dim)
            if data is not None:
                valid_graphs.append(data)
                weights.append(pattern_family_weight(pattern_item))
                fallback_weights.append(pattern_family_support(pattern_item))
                scores.append(pattern_family_score(pattern_item))

        prototype_counts[int(class_idx)] = len(valid_graphs)
        if valid_graphs:
            weight_tensor = torch.tensor(weights, dtype=torch.float)
            fallback_tensor = torch.tensor(fallback_weights, dtype=torch.float)
            unique_scores = {round(score, 12) for score in scores}

            if weight_tensor.sum().item() <= 0 or len(unique_scores) <= 1:
                weight_tensor = fallback_tensor if fallback_tensor.sum().item() > 0 else torch.ones_like(weight_tensor)

            prototype_bank[int(class_idx)] = {
                "graphs": Batch.from_data_list(valid_graphs),
                "weights": weight_tensor,
            }
        else:
            prototype_bank[int(class_idx)] = None

    return prototype_bank, prototype_counts

def parse_args():
    parser = argparse.ArgumentParser(description='Train MyExplainer model')

    # 基础设置
    parser.add_argument('--cuda', type=int, default=0, help='GPU device')
    parser.add_argument('--dataset', type=str, default='alkane_carbonyl', help='Dataset name')
    parser.add_argument('--gnn_path', type=str, default='param/', help='GNN directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--train_mode',type=bool,default=True,help='Current mode')
    parser.add_argument('--task', type=str, default='graph', help='Task type: graph classification or node classification')


    # 数据参数
    parser.add_argument('--top_k', type=int, default=1, help='Number of similar graphs for pairing')
    parser.add_argument('--threshold', type=float, default=0, help='Prediction confidence threshold')
    parser.add_argument(
        '--subgraph_sample_threshold',
        type=float,
        default=None,
        help='Subgraph sampling edge probability threshold; uses dataset-specific default when unset',
    )
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # 模型参数
    # ba2: 128,32    mutag:256,32
    # parser.add_argument('--x_dim', type=int, default=10, help='Node feature dimension (14 for mutag)')
    parser.add_argument('--h_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent dimension')

    parser.add_argument('--max_num_nodes', type=int, default=25, help='Maximum number of nodes in a graph')         # 53, 28
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument(
        '--loss_config_path',
        type=str,
        default=None,
        help='Path to dataset-specific loss hyperparameter YAML config'
    )
    parser.add_argument('--proto_topk', type=int, default=100, help='Top-K discriminative pattern families per class')
    parser.add_argument('--proto_refresh_every', type=int, default=5, help='Refresh prototypes every N epochs')
    parser.add_argument('--pattern_family_min_count', type=int, default=2,
                        help='Minimum generated samples needed to keep a pattern family candidate')
    parser.add_argument('--pattern_min_support', type=float, default=0.05,
                        help='Minimum in-class graph support required for a discriminative pattern family')

    parser.add_argument('--subgraph_method',type=str,default='genGraphEx',help='Subgraph method')

    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset = args.dataset.lower()
    apply_loss_hparams(args)

    # 设置设备为torch.device对象
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    print("Loss hyperparameters:")
    for key in LOSS_HPARAM_KEYS:
        print(f"  {key}: {getattr(args, key)}")

    # 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
    args.x_dim = train_dataset[0].x.shape[1]
    args.edge_attr_dim = train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0
    args.num_classes = 2
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载被解释GNN
    print("\n2. Loading pre-trained GNN classifier...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
    for p in gnn.parameters():
        p.requires_grad_(False)
    gnn.eval()
    print("  GNN loaded successfully")

    # 对训练集进行预测分类
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        for data in train_dataset:
            data = data.to(args.device)
            out = gnn(data.x, data.edge_index, data.batch)
            pred_probs.extend(out.softmax(dim=1))
            preds = out.argmax(dim=1).cpu()
            pred_labels.extend(preds)

    # 使用预测结果分别提取预测为0类和1类的数据集子集
    indices_0 = [i for i, pred in enumerate(pred_labels) if pred == 0]
    indices_1 = [i for i, pred in enumerate(pred_labels) if pred == 1]
    train_dataset_0, train_dataset_1 = train_dataset[indices_0], train_dataset[indices_1]
    splited_train_dataset = {0: train_dataset_0, 1: train_dataset_1}


    # 加载子图模式
    # 如果fsm_results/args.dataset_patterns.pkl不存在，则运行FSMiner进行挖掘,否则直接加载
    patterns = subgraph_mining(args,splited_train_dataset)
    prototype_bank, prototype_counts = build_prototype_bank(patterns, args.x_dim, args.proto_topk)
    for class_idx, count in prototype_counts.items():
        print(f"  Prototype graphs for class {class_idx}: {count}")



    print("\n4. Creating dataset with subgraph masks...")

    train_dataset_with_masks = MappedDataset(
        args, train_dataset, patterns, pred_labels, pred_probs, split_name="train"
    )
    test_dataset_with_masks = MappedDataset(
        args, test_dataset, patterns, gnn=gnn, split_name="test"
    )
    val_dataset_with_masks = MappedDataset(
        args, val_dataset, patterns, gnn=gnn, split_name="val"
    )

    print("\n5. Creating masked data loader...")
    train_loader_masked = TorchDataLoader(
        train_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn
    )
    test_loader_masked = TorchDataLoader(
        test_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn
    )
    val_loader_masked = TorchDataLoader(
        val_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(train_loader_masked)}")

    if args.train_mode:
        # 初始化模型
        print("\n7. Initializing MyExplainer model...")
        model = MyExplainerV2(args, gnn).to(args.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {num_params:,}")

        # 设置优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=15,
            verbose=True,
            min_lr=1e-6
        )
        print(f"  Optimizer: Adam")
        print(f"  Learning rate: {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")

        # 训练模型
        print("\n8. Training MyExplainer with subgraph masks...")
        print("=" * 80)

        trained_model, losses = train_myexplainerV2(
            args=args,
            model=model,
            gnn=gnn,
            train_loader=train_loader_masked,
            eval_loader=val_loader_masked,
            optimizer=optimizer,
            scheduler=scheduler,
            prototype_bank=prototype_bank,
            epochs=args.epochs
        )
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    else:
        print("\n8. Loading Trained MyExplainer...")
        print("=" * 80)
        trained_model = MyExplainerV2(args, gnn).to(args.device)
        trained_model.load_state_dict(
            torch.load(f'param/myexplainer_{args.dataset.lower()}_best.pt', map_location=args.device),
            strict=False
        )
        trained_model.refresh_class_prototypes(prototype_bank)
        trained_model.eval()
        for p in trained_model.parameters():
            p.requires_grad_(False)
        print("\n" + "=" * 80)
        print("Loading completed successfully!")
        print("=" * 80)



    # 8. 评估模型
    print("\n9. Evaluating on validation set...")
    print("=" * 80)

    evaluation_metrics = evaluate(
        args=args,
        model=trained_model,
        gnn=gnn,
        data_loader=val_loader_masked,
    )
    print("\nEvaluation Results on Validation Set:")
    print(
        "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
            evaluation_metrics["validity"],
            int(evaluation_metrics["successful"]),
            int(evaluation_metrics["total"]),
        )
    )
    print(
        "  Proximity ↓: {:.4f}".format(
            evaluation_metrics["proximity"]
        )
    )
    # print(
    #     "  Fidelity+ ↑: {:.4f}".format(
    #         evaluation_metrics["fidelity+"]
    #     )
    # )
    # print(
    #     "  Fidelity- ↓: {:.4f}".format(
    #         evaluation_metrics["fidelity-"]
    #     )
    # )
    print(
        "  Fidelity_prob ↑: {:.4f}".format(
            evaluation_metrics["fidelity"]
        )
    )
    print(
        "  Sparsity ↑: {:.4f}".format(
            evaluation_metrics["sparsity"]
        )
    )

    per_class_flip = evaluation_metrics.get("per_class_flip", {})
    for class_idx in [0, 1]:
        class_metrics = per_class_flip.get(class_idx, {})
        print(
            "  Class {} Flip Success -> {:.4f} (successful: {}/total: {}, target: {})".format(
                class_idx,
                class_metrics.get("ratio", 0.0),
                int(class_metrics.get("successful", 0)),
                int(class_metrics.get("total", 0)),
                1 - class_idx,
            )
        )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
