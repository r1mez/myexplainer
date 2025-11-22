
import argparse
from datetime import datetime
from xmlrpc.client import boolean
import os
import pickle
import hashlib

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from utils import get_datasets, GraphPairData, custom_collate_fn, GraphTrainData, train_collate_fn
from models.myexplainer import MyExplainer, MyExplainerBA2, MyCausalExplainer
from utils.pair_data import GraphTrainDataBA2
from utils.ps.mol_bpe import graph_bpe
from utils.train_myexplainer import train_myexplainer_with_subgraph, train_myexplainer_with_causality
from gnns import *

from utils.FSM.subgraph_mining.decoder import FSMiner
from evaluation import evaluate

import random
import numpy as np
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MyExplainer model')

    # 基础设置
    parser.add_argument('--cuda', type=int, default=1, help='GPU device')
    parser.add_argument('--dataset', type=str, default='ba2motif', help='Dataset name')
    parser.add_argument('--gnn_path', type=str, default='param/', help='GNN directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--train_mode',type=bool,default=True,help='Current mode')
    parser.add_argument('--task', type=str, default='graph', help='Task type: graph classification or node classification')

    # 数据参数
    parser.add_argument('--top_k', type=int, default=1, help='Number of similar graphs for pairing')
    parser.add_argument('--threshold', type=float, default=0, help='Prediction confidence threshold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    parser.add_argument("--loss_recon_x", type=float, default=0.0, help="Reconstruction loss weight for node features")
    parser.add_argument("--loss_recon_adj", type=float, default=1.0, help="Reconstruction loss weight for adjacency matrix")
    parser.add_argument("--loss_kl", type=float, default=0.0, help="KL divergence loss weight")
    parser.add_argument("--loss_pred", type=float, default=0.0, help="Prediction loss weight")
    parser.add_argument("--loss_fid", type=float, default=0.0, help="Fidelity loss weight")

    # 模型参数
    # parser.add_argument('--x_dim', type=int, default=10, help='Node feature dimension (14 for mutag)')
    parser.add_argument('--h_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=64, help='Latent dimension')

    parser.add_argument('--max_num_nodes', type=int, default=25, help='Maximum number of nodes in a graph')         # 53, 28
    parser.add_argument('--edge_attr_dim', type=int, default=0, help='Edge attribute dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    return parser.parse_args()


def main():
    args = parse_args()
    args.loss_proportion = {
        'recon_x': args.loss_recon_x,
        'recon_adj': args.loss_recon_adj,
        'kl': args.loss_kl,
        'pred': args.loss_pred,
        # 'fid': args.loss_fid
    }

    # 设置设备为torch.device对象
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    args.x_dim = train_dataset.num_features
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载被解释GNN
    print("\n2. Loading pre-trained GNN classifier...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
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
    print(pred_probs)



    # 加载子图模式
    # 如果fsm_results/args.dataset_patterns.pkl不存在，则运行FSMiner进行挖掘,否则直接加载
    print("\n3. Loading or mining frequent subgraph patterns...")
    patterns_0_path = f'fsm_results/{args.dataset}_0_patterns.pkl'
    patterns_1_path = f'fsm_results/{args.dataset}_1_patterns.pkl'
    if os.path.exists(patterns_0_path) and os.path.exists(patterns_1_path):
        print(f"  Found existing patterns, loading...")
        with open(patterns_0_path, 'rb') as f:
            patterns_0 = pickle.load(f)
        with open(patterns_1_path, 'rb') as f:
            patterns_1 = pickle.load(f)
        print(f"  Loaded {len(patterns_0 )} patterns for class 0 and {len(patterns_1)} patterns for class 1")
    else:
        print(f"  No existing patterns found, mining from training data...")
        FSMiner(train_dataset_0, 0)
        FSMiner(train_dataset_1, 1)
        print("  Finished mining frequent subgraph patterns.")
        with open(patterns_0_path, 'rb') as f:
            patterns_0 = pickle.load(f)
        with open(patterns_1_path, 'rb') as f:
            patterns_1 = pickle.load(f)

    # def reverse_groups_new(lst, group_size=3):
    #     """
    #     返回新列表，按组倒序。
    #     """
    #     n = len(lst) // group_size
    #     result = []
    #     for i in range(n - 1, -1, -1):  # 从后往前遍历组索引
    #         result.extend(lst[i * group_size:(i + 1) * group_size])
    #     return result
    # patterns_0 = reverse_groups_new(patterns_0, group_size=3)
    # patterns_1 = reverse_groups_new(patterns_1, group_size=3)
    patterns = {0: patterns_0, 1: patterns_1}



    print("\n4. Creating dataset with subgraph masks...")

    train_dataset_with_masks = GraphTrainDataBA2(args, train_dataset, patterns, pred_labels, pred_probs)
    val_dataset_with_masks = GraphTrainDataBA2(args, val_dataset, patterns,gnn=gnn)


    print("\n5. Creating masked data loader...")
    train_loader_masked = TorchDataLoader(
        train_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    val_loader_masked = TorchDataLoader(
        val_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(train_loader_masked)}")

    # 初始化模型
    print("\n7. Initializing MyExplainer model...")
    model = MyExplainerBA2(args, gnn).to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    # 训练模型
    print("\n8. Training MyExplainer with subgraph masks...")
    print("=" * 80)

    trained_model, losses = train_myexplainer_with_subgraph(
        args=args,
        model=model,
        gnn=gnn,
        train_loader=train_loader_masked,
        eval_loader=val_loader_masked,
        optimizer=optimizer,
        epochs=args.epochs
    )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

    # 8. 评估模型（使用原始验证集，不使用配对数据）
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
    print(
        "  Fidelity+ ↑: {:.4f}".format(
            evaluation_metrics["fidelity+"]
        )
    )
    print(
        "  Fidelity- ↓: {:.4f}".format(
            evaluation_metrics["fidelity-"]
        )
    )
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

    # evaluation_metrics = evaluate(
    #     args=args,
    #     model=trained_model,
    #     gnn=gnn,
    #     data_loader=train_loader_masked,
    # )
    # print("\nEvaluation Results on Training Set:")
    # print(
    #     "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
    #         evaluation_metrics["validity"],
    #         int(evaluation_metrics["successful"]),
    #         int(evaluation_metrics["total"]),
    #     )
    # )
    # print(
    #     "  Proximity ↓: {:.4f}".format(
    #         evaluation_metrics["proximity"]
    #     )
    # )
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
    # print(
    #     "  Fidelity_prob ↑: {:.4f}".format(
    #         evaluation_metrics["fidelity"]
    #     )
    # )
    # print(
    #     "  Sparsity ↑: {:.4f}".format(
    #         evaluation_metrics["sparsity"]
    #     )
    # )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


def main_with_causality():
    args = parse_args()
    args.loss_proportion = {
        'recon_x': args.loss_recon_x,
        'recon_adj': args.loss_recon_adj,
        'kl': args.loss_kl,
        'pred': args.loss_pred,
        'ortho':0,
    }

    # 设置设备为torch.device对象
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    args.x_dim = train_dataset[0].x.shape[1]
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载被解释GNN
    print("\n2. Loading pre-trained GNN classifier...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
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
    print(pred_probs)



    # 加载子图模式
    # 如果fsm_results/args.dataset_patterns.pkl不存在，则运行FSMiner进行挖掘,否则直接加载
    print("\n3. Loading or mining frequent subgraph patterns...")
    patterns_0_path = f'fsm_results/{args.dataset}_0_patterns.pkl'
    patterns_1_path = f'fsm_results/{args.dataset}_1_patterns.pkl'
    if os.path.exists(patterns_0_path) and os.path.exists(patterns_1_path):
        print(f"  Found existing patterns, loading...")
        with open(patterns_0_path, 'rb') as f:
            patterns_0 = pickle.load(f)
        with open(patterns_1_path, 'rb') as f:
            patterns_1 = pickle.load(f)
        print(f"  Loaded {len(patterns_0 )} patterns for class 0 and {len(patterns_1)} patterns for class 1")
    else:
        print(f"  No existing patterns found, mining from training data...")
        FSMiner(train_dataset_0, 0)
        FSMiner(train_dataset_1, 1)
        print("  Finished mining frequent subgraph patterns.")
        with open(patterns_0_path, 'rb') as f:
            patterns_0 = pickle.load(f)
            patterns_0 = patterns_0[::-1]
        with open(patterns_1_path, 'rb') as f:
            patterns_1 = pickle.load(f)

    # def reverse_groups_new(lst, group_size=3):
    #     """
    #     返回新列表，按组倒序。
    #     """
    #     n = len(lst) // group_size
    #     result = []
    #     for i in range(n - 1, -1, -1):  # 从后往前遍历组索引
    #         result.extend(lst[i * group_size:(i + 1) * group_size])
    #     return result
    # patterns_0 = reverse_groups_new(patterns_0, group_size=3)
    # patterns_1 = reverse_groups_new(patterns_1, group_size=3)
    patterns = {0: patterns_0, 1: patterns_1}



    print("\n4. Creating dataset with subgraph masks...")

    train_dataset_with_masks = GraphTrainDataBA2(args, train_dataset, patterns, pred_labels, pred_probs)
    val_dataset_with_masks = GraphTrainDataBA2(args, val_dataset, patterns,gnn=gnn)


    print("\n5. Creating masked data loader...")
    train_loader_masked = TorchDataLoader(
        train_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    val_loader_masked = TorchDataLoader(
        val_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(train_loader_masked)}")

    # 初始化模型
    print("\n7. Initializing MyExplainer model...")
    model = MyCausalExplainer(args, gnn).to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    # 训练模型
    print("\n8. Training MyExplainer with subgraph masks...")
    print("=" * 80)

    trained_model, losses = train_myexplainer_with_causality(
        args=args,
        model=model,
        gnn=gnn,
        train_loader=train_loader_masked,
        eval_loader=val_loader_masked,
        optimizer=optimizer,
        epochs=args.epochs
    )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

    # 8. 评估模型（使用原始验证集，不使用配对数据）
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
    print(
        "  Fidelity+ ↑: {:.4f}".format(
            evaluation_metrics["fidelity+"]
        )
    )
    print(
        "  Fidelity- ↓: {:.4f}".format(
            evaluation_metrics["fidelity-"]
        )
    )
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

    # evaluation_metrics = evaluate(
    #     args=args,
    #     model=trained_model,
    #     gnn=gnn,
    #     data_loader=train_loader_masked,
    # )
    # print("\nEvaluation Results on Training Set:")
    # print(
    #     "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
    #         evaluation_metrics["validity"],
    #         int(evaluation_metrics["successful"]),
    #         int(evaluation_metrics["total"]),
    #     )
    # )
    # print(
    #     "  Proximity ↓: {:.4f}".format(
    #         evaluation_metrics["proximity"]
    #     )
    # )
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
    # print(
    #     "  Fidelity_prob ↑: {:.4f}".format(
    #         evaluation_metrics["fidelity"]
    #     )
    # )
    # print(
    #     "  Sparsity ↑: {:.4f}".format(
    #         evaluation_metrics["sparsity"]
    #     )
    # )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main_with_causality()
