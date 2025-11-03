"""
MyExplainer训练示例脚本
展示如何使用train_myexplainer训练函数
"""

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
from models.myexplainer import MyExplainer
from utils.ps.mol_bpe import graph_bpe
from utils.train_myexplainer import train_myexplainer, evaluate_myexplainer, train_myexplainer_with_subgraph
from gnns import *

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
    parser.add_argument('--cuda', type=int, default=2, help='GPU device')
    parser.add_argument('--dataset', type=str, default='mutag', help='Dataset name')
    parser.add_argument('--gnn_path', type=str, default='param/', help='GNN directory')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (cpu or cuda)')
    parser.add_argument('--train_mode',type=bool,default=True,help='Current mode')

    # 数据参数
    parser.add_argument('--top_k', type=int, default=1, help='Number of similar graphs for pairing')
    parser.add_argument('--threshold', type=float, default=0, help='Prediction confidence threshold')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument("--loss_recon_x", type=float, default=1.0, help="Reconstruction loss weight for node features")
    parser.add_argument("--loss_recon_adj", type=float, default=5.0, help="Reconstruction loss weight for adjacency matrix")
    parser.add_argument("--loss_diversity", type=float, default=0.0, help="Diversity loss weight")
    parser.add_argument("--loss_distribution", type=float, default=0.0, help="Distribution matching loss weight")
    parser.add_argument("--loss_kl", type=float, default=1.0, help="KL divergence loss weight")
    parser.add_argument("--loss_pred", type=float, default=5.0, help="Prediction loss weight")

    # 模型参数
    parser.add_argument('--x_dim', type=int, default=14, help='Node feature dimension (14 for mutag)')
    parser.add_argument('--h_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--u_dim', type=int, default=32, help='Graph feature dimension')
    parser.add_argument('--edge_attr_dim', type=int, default=3, help='Edge attribute dimension (3 for mutag)')
    parser.add_argument('--max_num_nodes', type=int, default=53, help='Maximum number of nodes in a graph')         # 53, 28
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    return parser.parse_args()


def main1():
    args = parse_args()

    # 设置设备
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 1. 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 2. 加载预训练的GNN模型
    print("\n2. Loading pre-trained GNN...")
    gnn = torch.load(f'{args.gnn_path}gnns/{args.dataset.lower()}_gcn.pt',
                     map_location=args.device)
    gnn.eval()
    print("  GNN loaded successfully")

    # 3. 创建图对数据集
    print("\n3. Creating graph pair datasets...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 先检测数据集中的实际最大节点数
    print("  Detecting actual max_num_nodes from dataset...")
    all_datasets = list(train_dataset) + list(val_dataset) + list(test_dataset)
    actual_max_nodes = max([data.num_nodes for data in all_datasets])
    print(f"  Actual max nodes in dataset: {actual_max_nodes}")

    # 如果用户设置的max_num_nodes小于实际值，更新它
    if args.max_num_nodes < actual_max_nodes:
        print(f"  WARNING: Configured max_num_nodes ({args.max_num_nodes}) < actual max ({actual_max_nodes})")
        args.max_num_nodes = actual_max_nodes + 2  # 添加小余量
        print(f"  Updated max_num_nodes to: {args.max_num_nodes}")

    train_paired_dataset = GraphPairData(args, train_loader, gnn, k=args.top_k)
    val_paired_dataset = GraphPairData(args, val_loader, gnn, k=args.top_k)

    print(f"  Train pairs: {len(train_paired_dataset)}")
    print(f"  Val pairs: {len(val_paired_dataset)}")

    # 4. 创建数据加载器
    print("\n4. Creating data loaders...")
    train_pair_loader = DataLoader(
        train_paired_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 5. 初始化MyExplainer模型
    print("\n5. Initializing MyExplainer model...")
    model = MyExplainer(args, gnn).to(args.device)

    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # 6. 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 7. 训练模型
    print("\n6. Training MyExplainer...")
    print("=" * 80)

    trained_model, losses = train_myexplainer(
        args=args,
        model=model,
        gnn=gnn,
        pair_loader=train_pair_loader,
        eval_loader=val_eval_loader,
        optimizer=optimizer,
        epochs=args.epochs
    )

    # 8. 评估模型（使用原始验证集，不使用配对数据）
    print("\n7. Evaluating on validation set...")
    print("=" * 80)


    val_metrics = evaluate_myexplainer(
        args=args,
        model=trained_model,
        gnn=gnn,
        val_loader=val_eval_loader  # 使用原始验证集
    )

    # 9. 保存最终模型
    print("\n8. Saving final model...")
    torch.save({
        'args': args,
        'model_state_dict': trained_model.state_dict(),
        'losses': losses,
        'val_metrics': val_metrics
    }, f'{args.gnn_path}myexplainer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    print(f"  Model saved to {args.gnn_path}myexplainer_best.pt")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

def main():
    args = parse_args()
    args.loss_proportion = {
        'recon_x': args.loss_recon_x,
        'recon_adj': args.loss_recon_adj,
        'diversity': args.loss_diversity,
        'distribution': args.loss_distribution,
        'kl': args.loss_kl,
        'pred': args.loss_pred
    }

    # 设置设备为torch.device对象
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # 检测实际最大节点数
    print("\n2. Detecting actual max_num_nodes from dataset...")
    all_datasets = list(train_dataset) + list(val_dataset) + list(test_dataset)
    actual_max_nodes = max([data.num_nodes for data in all_datasets])
    print(f"  Actual max nodes in dataset: {actual_max_nodes}")

    if args.max_num_nodes < actual_max_nodes:
        print(f"  WARNING: Configured max_num_nodes ({args.max_num_nodes}) < actual max ({actual_max_nodes})")
        args.max_num_nodes = actual_max_nodes + 2
        print(f"  Updated max_num_nodes to: {args.max_num_nodes}")

    # 提取频繁子图词汇表
    print("\n3. Extracting frequent subgraph vocabulary...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # smiles_0=[]
    # smiles_1=[]
    # for data in train_loader:
    #     data.to(args.device)
    #     # 获取预测结果并确定每个样本的类别
    #     pred_labels = gnn.get_pred(data.x, data.edge_index, data.batch)[0].argmax(dim=1)
    #
    #     # 分别处理每个样本
    #     for i in range(len(pred_labels)):
    #         if pred_labels[i] == 0:
    #             smiles_0.append(data.smiles[i] if isinstance(data.smiles, list) else data.smiles)
    #         else:
    #             smiles_1.append(data.smiles[i] if isinstance(data.smiles, list) else data.smiles)
    #
    # with open(f"data/{args.dataset}/smiles/smiles_0.txt", "w") as f:
    #     for smi in smiles_0:
    #         print(smi, file=f)
    # with open(f"data/{args.dataset}/smiles/smiles_1.txt", "w") as f:
    #     for smi in smiles_1:
    #         print(smi, file=f)

    # vocab_len = 100
    #
    # smis_0, _ = graph_bpe(f'data/{args.dataset}/smiles/smiles_0.txt', vocab_len,
    #                       f'data/mutag/smiles_bpe_{vocab_len}_0.txt', 16, False)
    # smis_1, _ = graph_bpe(f'data/{args.dataset}/smiles/smiles_1.txt', vocab_len,
    #                       f'data/mutag/smiles_bpe_{vocab_len}_1.txt', 16, False)


    # Mutagenicity 500
    # smis_0 = ['COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)CC3OC1CCC(O)C(C)O1',
    #          'CC1OC(OC2CC(O)Cc3c(O)c4c(c(O)c32)C(=O)c2ccccc2C4=O)CCC1O',
    #          'Nc1cc(S(=O)(=O)O)cc2cc(S(=O)(=O)O)c(N=Nc3ccccc3)c(O)c12',
    #          'N=Nc1ccc(-c2ccc(N=Nc3ccc(O)c(C(=O)O)c3)cc2)cc1', 'COc1ccc(O)c2c(=O)c3c(OC)cc4c(c3oc12)C1CCOC1O4',
    #          'CN(C)CCCNc1c2ccccc2nc2cccc([N+](=O)[O-])c12', 'O=C1c2ccccc2C(=O)c2c(O)c3c(c(O)c21)CC(O)CC3O',
    #          'O=[N+]([O-])c1ccc2c3cccc4cccc(c5cccc1c52)c43', 'O=c1c2[nH]c3ccccc3c2nnn1N=Cc1ccccc1',
    #          'c1ccc2c(c1)ccc1[nH]c3ccc4ccccc4c3c12', 'COc1cc(OC)c2c(=O)c3c(O)ccc(OC)c3oc2c1',
    #          'Nc1c(S(=O)(=O)O)cc2cc(S(=O)(=O)O)cc(N)c2c1O', 'Nc1ccc2c(S(=O)(=O)O)cc(S(=O)(=O)O)c(N)c2c1O',
    #          'O=C1c2ccccc2C(=O)c2c(O)c(CCO)cc(O)c21', 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
    #          'COc1ccc(O)c2c(=O)c3c(OC)cccc3oc12', 'cccc1cccc2cccc(cccc[N+](=O)[O-])c12',
    #          'O=[N+]([O-])c1occ(-c2ccccc2)c1-c1ccccc1', 'O=[N+]([O-])c1ccc2c([N+](=O)[O-])cc3ccccc3c2c1',
    #          'OC1C=Cc2c(ccc3c2ccc2ccccc23)C1O', 'O=[N+]([O-])c1ccc2ccc3cccc4ccc1c2c34', 'Cc1c2ccccc2cc2ccc3ccccc3c12',
    #          'Oc1ccc2ccccc2c1N=Nc1ccccc1', 'O=C1c2ccccc2C(=O)c2c1cccc2[N+](=O)[O-]',
    #          'Nc1ncnc2c1ncn2-c1ccc([N+](=O)[O-])cc1', 'Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O',
    #          'CN1C(=O)CN=C(c2ccccc2)c2ccccc21', 'c1ccc2c(c1)ccc1c3ccccc3ccc21', 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
    #          'O=C1c2cccc(O)c2C(=O)c2c(O)cccc21', 'O=C1c2ccccc2C(=O)c2c(O)ccc(O)c21', 'COc1cccc2oc3cccc(O)c3c(=O)c12',
    #          'O=c1c2c(O)cccc2oc2cc(O)cc(O)c12', 'c1ccc2nc3ccc4ccccc4c3cc2c1', 'COc1cc(-c2ccc(N)c(OC)c2)ccc1N',
    #          'Nc1ccc(O)c2c1C(=O)c1ccccc1C2=O', 'O=C(O)c1cn(C2CC2)c2ccc(F)cc2c1=O', 'O=C1c2ccccc2C(=O)c2c(O)cccc21',
    #          'O=[N+]([O-])c1ccc2ccc3ccccc3c2c1', 'O=C(Nc1ccccc1)c1csc([N+](=O)[O-])c1', 'Nc1cccc2c1C(=O)c1ccccc1C2=O',
    #          'O=[N+]([O-])c1ccc(C=Cc2ccccc2)cc1', 'O=[N+]([O-])c1ccc2c(ccc3ccccc32)c1', 'CC(=O)Nc1ccc2c(c1)Cc1ccccc1-2',
    #          'CCn1cc(C(=O)O)c(=O)c2cc(F)ccc21', 'O=c1c2c(O)cccc2oc2cccc(O)c12', 'Nc1ccc2cc(S(=O)(=O)O)cc(N)c2c1O',
    #          'O=[N+]([O-])c1cccc2nc3ccccc3cc12', 'CN(C)c1ccc(N=Nc2ccccc2)cc1', 'Nc1ccc2c(S(=O)(=O)O)ccc(N)c2c1O','O=[N+]([O-])c1cc2ccccc2c2ccccc12', 'c1ccc(-c2cocc2-c2ccccc2)cc1', 'O=[N+]([O-])c1c2ccccc2cc2ccccc12','Cc1c(C)c2c(nc(N)n2C)c2nccnc12', 'O=[N+]([O-])c1cccc2c1ccc1ccccc12', 'O=C1c2ccccc2C(=O)c2ccccc21','c1cc2ccc3cccc4ccc(c1)c2c34', 'O=c1c2ccccc2oc2cccc(O)c12', 'Cc1cc2c(nc(N)n2C)c2nccnc12','cccc1ccc2cccc3cccc1c32', 'Cc1cc2nccnc2c2nc(N)n(C)c12', 'Cc1cc(-c2ccc(N)c(C)c2)ccc1N', 'Cc1ccc(C)c2c1[nH]c1ccccc12', 'Cc1c2ccccc2c(C)c2ccccc12', 'CC(C)NC(=O)C=Cc1ccc([N+](=O)[O-])o1','N=Nc1ccc(-c2ccc(N=N)cc2)cc1', 'Nc1ccc2cc(S(=O)(=O)O)cc(O)c2c1', 'OC1C=Cc2cc3ccccc3cc2C1O','Oc1cccc2oc3ccccc3cc12', 'O=C(O)c1cnc2ccc(F)cc2c1=O', 'Cc1c2ccccc2cc2ccccc12','c1ccc2c(c1)-c1ccccc1C1NC21', 'Nc1ccc2cc3ccccc3nc2c1', 'O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1','Cc1nccc2[nH]c3ccccc3c12', 'CCNC(=O)C=Cc1ccc([N+](=O)[O-])o1', 'O=c1nnnc2c1[nH]c1ccccc12','Cc1cccc2c1ccc1ccccc12', 'Cc1cccc2[nH]c3ccccc3c12', 'C[n+]1c2ccccc2cc2ccccc21', 'Nc1c2ccccc2nc2ccccc12','O=c1c2ccccc2oc2ccccc12', 'O=[N+]([O-])c1ccc(c2ccccc2)cc1', 'Cc1cccc2ccc3ccccc3c12','Oc1ccc2ccc3ccccc3c2c1', 'c1ccc2c(c1)ccc1ccccc12', 'c1ccc2nc3ccccc3cc2c1', 'c1ccc2cc3ccccc3cc2c1','c1ccc2c(c1)[nH]c1ccccc12', 'Oc1ccccc1cc1ccccc1', 'c1ccc2c(c1)[nH]c1ccncc12', 'c1ccc(N=Nc2ccccc2)cc1','Cn1cnc2c3cccnc3ccc21', 'c1ccc2c(c1)[nH]c1cnnnc12', 'c1ccc2c(c1)CCc1ccccc1-2', 'Nc1ccc2c(c1)Cc1ccccc1-2','COC(=O)C=Cc1ccc([N+](=O)[O-])o1', 'COc1ccc2c(c1)OC1OC=CC21', 'O=c(c1ccccc1)c1ccccc1','O=c1c(O)coc2cc(O)cc(O)c12', 'Nc1ccc(-c2ccc(N)cc2)cc1', 'Nc1ccc(Oc2ccccc2)cc1', 'COc1cc2ccncc2cc1OC','c1ccc2c(c1)Cc1ccccc1-2', 'cccc1cccc2ccccc12', 'c1ccc(cc2ccccc2)cc1', 'Nc1ccc2cccc(N)c2c1O','Nc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]', 'O=[N+]([O-])c1ccc2ccccc2c1', 'Cc1c([N+](=O)[O-])cccc1[N+](=O)[O-]','c1ccc2c(c1)oc1ccccc12', 'O=c1ccoc2cc(O)cc(O)c12', 'CNc1ccc2nccnc2c1C', 'O=[N+]([O-])c1cccc2ccccc12','c1ccc(Nc2ccccc2)cc1', 'COc1cccc2oc(=O)ccc12', 'Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]','O=[N+]([O-])c1ccccc1SSCCl', 'Nc1ccc(-c2ccccc2)cc1', 'ccccccc1ccccc1','O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'c1ccc(c2ccccc2)cc1', 'ccc1ccc2ccccc2c1', 'Nc1ccc2ccccc2c1O','c1ccc(-c2ccccc2)cc1', 'ccc1cccc2ccccc12', 'c1ccc(c2cccnc2)cc1', 'O=c1ccnc2ccc(F)cc12','Nc1c(O)ccc2ccccc12', 'CCOCCOCCOCCO', '[nH]c1ccc2ccccc2c1', 'Oc1cc(O)c2cccoc2c1', 'c1ccc(c2ccnnn2)cc1',           'OCC1OC(O)C(O)C(O)C1O', 'Cc1cccc2nc(N)n(C)c12', 'C1=Cc2cccc3cccc1c23', 'Nc1ccc2ccccc2c1N',           'O=CC=Cc1ccc([N+](=O)[O-])o1', 'O=Cc1ccc2[nH]cnc2c1', 'COc1ccc2c(c1)OCC2C', 'cc1ccc2ccccc2c1c',           'CS(=O)(=O)Nc1ccc(N)cc1', 'CC(=O)c1ccc([N+](=O)[O-])cc1', 'ccccc1ccccc1cc', 'Cc1ccc(C)c2ccccc12',           'Nc1ccc2cccc(O)c2c1', 'COc1ccc2ccncc2c1', 'nnc1c[nH]c2ccccc12', 'Nc1ccc2ccccc2c1', 'cccccccc[N+](=O)[O-]',           'Nc1cccc2ccccc12', 'Cc1cccc2nccnc12', 'Cc1ccc2ccccc2c1', 'OCC1OCC(O)C(O)C1O', 'Cc1cccc2ccccc12',            'CCc1cccc([N+](=O)[O-])c1', 'Fc1ccc2ncccc2c1', 'O=c1ccc2ccccc2o1', 'O=C1NC(=O)c2ccccc21',            'c1ccc(-c2ccoc2)cc1', 'Oc1ccc2cccoc2c1', 'Cc1cnc2ccccc2n1', 'cccc(C)c1ccccc1', 'cc1ccc2ccccc2c1',            'COc1nsc2ccccc12', 'Nc1ccc([N+](=O)[O-])cc1Cl', 'CCO[PH](=O)OC(C)CCl', 'cc1cccc2ccccc12',            'O=Cc1cccc([N+](=O)[O-])c1', 'O=Nn1ccc2ccccc21', 'O=CCN=Cc1ccccc1', 'CCNC(=O)N(CCCl)N=O',            'c1ccc(OCC2CO2)cc1', 'NC(=O)c1csc([N+](=O)[O-])c1', 'COc1cccc([N+](=O)[O-])c1', 'c1ccc2ccccc2c1','c1ccc2ncccc2c1', 'ccccc1ccccc1', 'Cc1ccccc1[N+](=O)[O-]', 'c1ccc2[nH]ccc2c1', 'Nc1ccc([N+](=O)[O-])cc1', 'c1ccc2occcc2c1', 'Cc1ccc([N+](=O)[O-])cc1', 'c1ccc2cnccc2c1', 'c1ccc2[nH]cnc2c1', 'O=[N+]([O-])c1ccc(O)cc1', 'Nc1ncnc2ncnc12', 'Nc1ccccc1[N+](=O)[O-]', 'COc1cccc(C=O)c1', 'CCc1ccc(OC)cc1', 'nc1cccc([N+](=O)[O-])c1', 'O=[N+]([O-])c1cccc(O)c1', 'O=C(O)c1ccccc1O', 'CCOc1ccc(N)cc1', 'CN(C)c1ccc(N)cc1', 'CC(=O)Nc1ccccc1', 'O=[N+]([O-])c1ccccc1S', 'OCC1OCC(O)CC1O', 'CCN(CCCl)CCCN', 'c1ccc2[nH]ncc2c1', 'CC1OCC(O)C(O)C1O', 'COc1ccccc1OC', 'OCCOc1ccccc1', 'Cn1cnc2ccccc21', 'Clc1nsc2ccccc12', 'N=Cc1ccc([N+](=O)[O-])o1', 'c1nc2c[nH]cnc-2n1', 'O=[N+]([O-])c1ccccc1', 'cccc1ccccc1', 'Cc1ccc([N+](=O)[O-])o1', 'c1ccc2sncc2c1', 'c1ccc(C2CN2)cc1', 'O=CNc1ccccc1', 'O=C(O)c1ccccc1', 'CN(C)c1ccccc1', 'COc1ccccc1N', 'c1ccc(C2CO2)cc1', 'COc1ccccc1O', 'cc1ccccc1cC', 'Cc1ccc(N)cc1N', 'c1ccc2nccc2c1', 'OCC1OCC(O)C1O', 'Cc1cc(N)ccc1N', 'ccccccccC', 'O=C(Cl)c1ccccc1', 'OC1COCC(O)C1O', 'c1ncc2ncnc2n1', 'cccccccc=O', 'nncc1ccccc1', 'NC(=O)c1ccccc1', 'O=Cc1ccccc1O', 'CCO[PH](=O)OCC', 'c1ccc2scnc2c1', 'CCNC(=O)CNC=O', 'CC(=O)c1ccccc1', 'ccccccccn', 'cccc1ccccn1', 'Cc1c(N)cccc1N', 'Cc1cc(n)ccc1N', 'O=Cc1cccc(O)c1', 'Cc1ccc(N)c(C)c1', 'COc1ccc(C)cc1', 'CCNc1ccccc1', 'ccccc(C)ccc', 'Cn1cncc1[N+](=O)[O-]', 'CCNC(=O)NCCCl', 'CCO[PH](=S)OCC', 'cccccccc', 'ccc1ccccc1', 'O=Cc1ccccc1', 'Cc1ccccc1N', 'CCc1ccccc1', 'CNc1ccccc1', 'ncc1ccccc1', 'Cc1ccc(N)cc1', 'N=Nc1ccccc1', 'O=[N+]([O-])c1cccs1', 'COc1ccccc1', 'cc1ccccc1c', 'cc1ccccc1n', 'Nc1ccc(N)cc1', 'N=Cc1ccccc1', 'Cc1cccc(O)c1', 'ClCc1ccccc1', '[nH]c1ccccc1', 'CC1OCCCC1O', 'Nc1ccccc1O', 'OCc1ccccc1', 'Cc1cccc(N)c1', 'Nc1ccc(O)cc1', 'oc1cccc(O)c1', 'CN(C)CCNC=O', 'O=cc1ccccc1', 'Nc1ccccc1Cl', 'nc1cccc(N)c1', 'cnc1ccccc1', 'OCC(O)C(O)CO', 'Oc1ccccc1O', 'COC(C=O)=CC=O', 'O=[N+]([O-])c1cncn1', 'O=C1OC(O)C=C1Cl', 'Nc1ccccc1N', 'Cc1ccc(F)cc1', '[O-][n+]c1ccccc1', 'Oc1ccc(Cl)cc1', 'OCC1OCCC1O', 'O=[N+]([O-])c1cncs1', 'Oc1ccc(O)cc1', 'O=[N+]([O-])c1ccco1', 'Cc1ccccc1', 'Nc1ccccc1', 'Oc1ccccc1', 'cc1ccccc1', 'CCOPOCC', 'ClCCNCCCl', 'OCCC(O)CO', 'Clc1ccccc1', 'Nc1ccncn1', 'nc1ccccc1', 'OCC(O)C1CO1', 'CN(C)CCCN', 'CCNCCCN', 'CN1CCNCC1', 'cc1ncnc1n', 'nc1cccnc1', 'Cc1cccnc1', 'CO[PH](=O)OC', 'Cc1ccccn1', 'N=C1C=CCC=C1', 'Fc1ccccc1', 'NCCNCCN', 'CO[PH](=S)OC', 'Nc1ccccn1', 'O=C1CCC(=O)N1', 'OC1C=COC1O', 'ClCC(Cl)=C(Cl)Cl', 'CCCOC(C)=O', 'CCCN(C)N=O', 'c1ccccc1', 'c1ccncc1', 'cccccc', 'CCOCCO', 'Cc1ccco1', 'c1cncnc1', 'CCNCCN', 'CCNC(C)=O', 'C1COCCN1', 'C1=CCC=CC1', 'C=CC(O)CO', 'CNCCCN', 'c1c[nH]cn1', 'CC(N)C(=O)O', 'Cn1ccnc1', 'CCOC(C)=O', 'O=CC=CC=O', 'OCC(O)CO', 'OC1CC=CO1', 'OC1CCOC1', 'C1CNCCN1', 'O=C1NCCO1', 'O=CNCCCl', 'cccccn', 'ccccCBr', 'CN(C=O)N=O', 'cncc[nH]', 'OCC(Br)CBr', 'O=NNCCO', 'cccccC', 'ccccC', 'O=[SH](=O)O', 'c1cncn1', 'c1ccsc1', 'CC(O)CO', 'CCNCC', 'Cnc(n)N', 'CCNC=O', 'ccccn', 'O=C(O)CCl', 'c1cscn1', 'c1ccoc1', 'CCC(C)O', 'C[SH](=O)=O', 'COPOC', 'CC(=O)NO', 'CNC(C)=O', 'CCNN=O', 'COC(C)=O', 'ccccc', 'Cc(n)cn', 'CCOP=O', 'NC(=O)CBr', 'CCOCC', 'CCC(C)C', 'OCC1CO1', 'COCC=O', 'CCCNC', 'ClCC(Cl)Cl', 'CC(C)(C)O', 'C1=COCC1', 'C=CC(=O)O', 'COC(N)=O', 'CCC(=O)O', 'OCCCBr', 'ClCCCCl', 'CNCCCl', 'ClCCCBr', 'CNC(=O)O', 'C1CSCN1', 'CCC1CO1', 'CC(C)CO', 'C1CCCC1', 'OCC(Cl)Cl', 'ClCC1CO1', 'COcns', 'O=C(O)CBr', 'NC(=O)CCl', 'cccc', 'CCNC', 'CC1CO1', 'cccn', 'CC(N)=O', 'CC(=O)O', 'OCCO', 'CCCO', 'CCCC', 'CCOP', 'Cncn', 'ccnc', 'CC(C)O', 'nccn', 'NCCCl', 'O=CCO', 'O=CCCl', 'cncn', 'nc[nH]', 'CNN=O', 'OCCCl', 'ClCCCl', 'NCC=O', 'CCC=O', 'NCCO', 'CC(C)N', 'C=CC=O', 'CNC=O', 'OCCBr', 'C=CCC', 'CCCCl', 'ClC(Cl)Cl', 'CC(C)C', 'CC(=O)Cl', 'CC(Cl)Cl', 'SCCCl', 'O=CCBr', 'Cccn', 'BrCCBr', 'CC(C)Br', 'CN(C)N', 'O=CC=O', 'COC=O', 'CCO', 'O=[N+][O-]', 'CCN', 'ccn', 'CC=O', 'CCC', 'C1CO1', 'CCCl', 'O=S=O', 'CNC', 'ncn', 'NC=O', 'O=CO', 'ClCCl', 'COP', 'ccc', 'CCBr', 'OCO', 'SCCl', 'ncN', 'N=CN', 'nc=O', 'CC[N+]', 'cco', 'C[S+]C', 'C=CC', 'cns', 'NCN', 'ccs', 'cc', 'CC', '[N+][O-]', 'cn', 'CO', 'CN', 'O=S', 'CCl', 'N=O', 'NO', 'C=O', 'NN', 'CS', '[n+][O-]', '[N+]=[N-]', 'C#N', 'C[n+]', 'C[S+]', 'c=O', 'CBr', 'C[N+]', 'nn', 'C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I']
    # smis_1 = ['CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)CC(C)C(O)C1(C)O',
    #          'CCCC(C)(O)C(OC1OC(C)CC(N(C)C)C1O)C(C)C(OC1CC(C)(OC)C(O)C(C)O1)C(C)C(=O)OC(CC)C(C)(O)CO',
    #          'CCCC(C)(O)C(OC1OC(C)CC(N(C)C)C1O)C(C)C(OC1CC(C)(OC)C(O)C(C)O1)C(C)C(=O)O',
    #          'CC1CCC2C(C)C3C(CC4C5CC=C6CC(O)CCC6(C)C5CCC43C)N2C1',
    #          'CCCC(C)(O)C(OC1OC(C)CC(N(C)C)C1O)C(C)C(O)C(C)C(=O)O', 'CCCCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C',
    #          'CC(CCC(=O)O)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C', 'CCCCCC(C)C1CCC2C3CC=C4CCCCC4(C)C3CCC12C',
    #          'CCNC(=O)Nc1ccc(OCC(O)CNC(C)(C)C)c(C(C)=O)c1', 'CC(=O)c1cc(NC(N)=O)ccc1OCC(O)CNC(C)(C)C',
    #          'C=C1CC23CCC4C(C)(C(=O)O)CCCC4(C)C2CCC1(O)C3', 'CCCCCC(C)C1CCC2C(CC)C(C(C)C)CCC12C',
    #          'CC(COC(=O)c1ccccc1)COC(=O)c1ccccc1', 'CCCN1c2ccccc2Sc2ccc(C(F)(F)F)cc21', 'CCCCCCCCOP(=O)(OCCCC)OCCCC',
    #          'CCC(O)CCCCCCCCCC(C)CCC(=O)O', 'CC(=O)c1ccccc1OCC(O)CNC(C)(C)C', 'CC(=O)Nc1ccc(Cc2cccc(Cl)c2)cc1Cl',
    #          'CC1CC2c3cccc4ncc(c34)CC2N(C)C1', 'O=C(O)c1ccccc1C(=O)OC1CCCCC1', 'FC(F)(F)c1ccc2c(c1)Nc1ccccc1S2',
    #          'CCCCCC(C)C(C)C(C)CCCC(C)C', 'cc1c2ccccc2C2CC(C)CN(C)C2C1', 'O=c1ccn(C2CC(O)C(CO)O2)c(=O)[nH]1',
    #          'CCCC(C)(O)C(O)C(C)C(O)C(C)C(=O)O', 'CCCCC(CC)COC(=O)c1ccccc1', 'CC=CC(C)=CC=CC=C(C)C=CC=C(C)C',
    #          'C=CCCCCC1C(C)CCCC1(C)C(=O)O', 'CC(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]',
    #          'CCCCOP(=O)(OCCCC)OCCCC', 'CCCCCCCCCCCCCCCC', 'CC(C)(C)NCC(O)COc1ccccc1', 'O=C(C=Cc1ccccc1)c1ccccc1',
    #          'CCC1(CC(=O)O)OCCc2cc[nH]c21', 'Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1', 'CCCCCCCCC(C)CCC(=O)O',
    #          'O=C(O)CCc1c[nH]c2ccccc12', 'Clc1cccc(Cc2cccc(Cl)c2)c1', 'c1ccc(OPOc2ccccc2)cc1',
    #          'O=S(=O)(O)c1ccc2cc(O)ccc2c1', 'O=S(=O)(O)c1ccc2ccccc2c1', 'CC=CC(C)=CC=CC=C(C)C=CC',
    #          'CCCCCC1C=CCC(C)(C)C1C', 'c1ccc2c(c1)CCc1ccccc1-2', 'c1ccc2c3c([nH]c2c1)CNCC3', 'CCCCCCCCCCCC(=O)O',
    #          'CC(CO)(CO[N+](=O)[O-])CO[N+](=O)[O-]', 'CC(=O)ON(C(C)=O)c1ccccc1', 'CC(C)COC(=O)c1ccccc1',
    #          'CC1=CC(C)(C)Nc2ccccc21', 'CCCCCCCCCCCCO', 'CC1(C)SC2CC(=O)N2C1C(=O)O', 'c1ccc(Oc2ccccc2)cc1',
    #          'COC(=O)C1C(C=C(C)C)C1(C)C', 'c1ccc(Nc2ccccc2)cc1', 'CC(=O)CC(=O)Nc1ccccc1', 'O=C(O)c1cnc2ncncc2c1',
    #          'CCCCO[PH](=O)OCCCC', 'CCCCCCCCCCCC', 'O=C(O)c1ccccc1C(=O)O', 'Cc1ccc(O)c(C(C)(C)C)c1',
    #          'CCCCCC1C=CC(=O)CC1', 'COc1ccc2ccccc2c1', 'O=c1ccoc2cccc(O)c12', 'CCCCC(CC)COC(C)=O',
    #          'CO[PH](=O)Oc1ccccc1', 'CCCCCCCCCC(=O)O', 'C=CCCCCCC(C)C(=O)O', 'N=Nc1ccc(S(=O)(=O)O)cc1',
    #          'OCC1OCC(O)C(O)C1O', 'CC(O)COc1ccccc1', 'Nc1ccc(S(N)(=O)=O)cc1', 'CC1CC(N(C)C)C(O)CO1', 'CCCCOPOCCCC',
    #          'NS(=O)(=O)c1ccccc1Cl', 'O=c1ccoc2ccccc12', 'CC=CC=CC=CC=CCC', 'CC(=O)Nc1ccccc1I', 'Nc1ncnc2[nH]cnc12',
    #          'Oc1ccc2ccccc2c1', 'CC1=C(C)C(C)(C)CCC1=O', 'COC1(C)CCOC(C)C1O', 'CCC(=O)Nc1ccccc1',
    #          'ClC1=C(Cl)C(Cl)C(Cl)(Cl)C1Cl', 'CCCCCCCCCCC', 'O=P(O)(O)Oc1ccccc1', 'O=c1ccc2ccccc2o1',
    #          'NCC(O)c1ccc(O)cc1', 'Cc1ccc(S(N)(=O)=O)cc1', 'Oc1cccc2ccccc12', 'COc1ccc(C)cc1OC', 'CCCCC(CC)COC=O',
    #          'CCCC(C)CCCC(C)C', 'Brc1cc(Br)c(Br)c(Br)c1Br', 'CC(C)Nc1ncnc(Cl)n1', 'Clc1cc(Cl)c(Cl)c(Cl)c1Cl',
    #          'CC(CO)(CO)CO[N+](=O)[O-]', 'CC1(C)CCCC(C)(C)N1O', 'OCC1OCC(O)CC1O', 'c1ccc2[nH]ccc2c1', 'CC(=O)Nc1ccccc1',
    #          'c1ccc2ccccc2c1', 'COC(=O)c1ccccc1', 'c1ccc2ncccc2c1', 'FC(F)(F)c1ccccc1', 'c1ccc2occcc2c1',
    #          'COc1ccccc1OC', 'CCCCCC(C)CCC', 'Clc1ccc(Cl)c(Cl)c1Cl', 'O=C(O)c1ccccc1O', 'C=C(C)C(=O)OCCCC',
    #          'CC1OCC(O)C(O)C1O', 'Nc1ccccc1C(=O)O', 'Brc1cc(Br)c(Br)c(Br)c1', 'CC1=C(C)C(C)(C)CCC1', 'CC(C)(O)c1ccccc1',
    #          'CC(C)Oc1ccccc1', 'COc1cc(C)ccc1O', 'CCCC1=CC(=O)CCC1', 'c1ncc2[nH]cnc2n1', 'c1ccc2nccnc2c1',
    #          'O=C(O)C1CCC(O)CO1', 'CC(C)Cc1ccccc1', 'O=C(O)CNC(=O)CCS', 'NC(=O)CCC(N)C(=O)O', 'C[N+](C)Cc1ccccc1',
    #          'ClC1=C(Cl)C(Cl)C(Cl)C1Cl', 'COC(=O)C1C(C)C1(C)C', 'Nc1ccc(C(=O)O)cc1', 'O=C(O)Cc1ccccc1',
    #          'O=Cc1cc(Br)cc(Br)c1', 'COc1ccc(C=O)cc1', 'CCCCOC(=O)CCC', 'O=c1cn[nH]c(=O)[nH]1', 'Oc1c(Br)cc(Br)cc1Br',
    #          'CC(=O)c1ccc(O)cc1', 'CCCCC(CC)COP', 'O=c1ncn(Cl)c(=O)n1Cl', 'O=C(O)c1ccccc1', 'FC(F)c1ccccc1',
    #          'CCCCCC(C)CC', 'CC(=O)c1ccccc1', 'CCCCC(CC)CO', 'Clc1ccc(Cl)c(Cl)c1', 'CCCCCCCCC', 'Oc1ccc(Cl)cc1Cl',
    #          'cccc1ccccc1', 'Oc1c(Br)cccc1Br', 'COc1ccc(C)cc1', 'NC(=O)c1ccccc1', 'CC=CC=C(C)C=CC', 'O=CNc1ccccc1',
    #          'CC1CCOC(C)C1O', 'CC1CCC(C)N(C)C1', 'CC1=CC(=O)C=CC1C', 'ClC1=C(Cl)C(Cl)C(Cl)C1', 'O=Cc1cccc(Br)c1',
    #          'CCCCCCC(C)C', 'C=CCCCCCCC', 'CCc1ccccc1N', 'O=c1ccnc(=O)[nH]1', 'O=Cc1cccc(Cl)c1', 'Brc1cc(Br)cc(Br)c1',
    #          'CC(CO)CO[N+](=O)[O-]', 'Clc1cccc(Cl)c1Cl', 'CCOC(=O)CC(C)C', 'OCCc1cc[nH]c1', 'O=C(O)CCCC(=O)O',
    #          'c1ccc2ocnc2c1', 'Cc1ccc(C)c(C)c1', 'C#CCCCCCCC', 'ClCc1cccc(Cl)c1', 'c1ccc2scnc2c1', 'CCC(O)C(C)C(=O)O',
    #          'O=c1ncn(Cl)c(=O)n1', 'O=C(O)c1cccnc1', 'O=C(O)C(F)C(F)CF', 'NC(=O)CCCC(=O)O', 'CC(=O)OCCCCO',
    #          'OCC1OCC(O)C1O', 'CCCc1ccccc1', 'Cc1ccc(O)c(C)c1', 'NC(=O)c1cccnc1', 'S=CNc1ccccc1', 'Cc1ccc(O)c(O)c1',
    #          'CN(C)c1ccccc1', 'FCCCCC(F)CF', 'CCC(C)COC(C)=O', 'CCCCCCC(C)=O', 'Oc1ccc(O)c(O)c1', 'OCCN1CCNCC1',
    #          'N=Nc1ccccc1Cl', 'CCCCCCC1CO1', 'NC(=O)c1cnccn1', 'CCCCCCCC', 'O=Cc1ccccc1', 'Clc1cccc(Cl)c1',
    #          'FCc1ccccc1', 'ccc1ccccc1', 'COc1ccccc1', 'Oc1ccccc1Br', 'Cc1ccc(O)cc1', 'CCc1ccccc1', 'Clc1ccccc1Cl',
    #          'Cc1cccc(C)c1', 'Brc1cccc(Br)c1', 'Cc1cccc(Cl)c1', 'OCC1OCCC1O', 'Oc1ccccc1O', 'Cc1ccccc1C',
    #          'O=c1ccnc[nH]1', 'cnc1ccccc1', 'CC1CCC(O)CO1', 'O=c1ncnc(=O)n1', 'CCCCC=C(C)C', 'CCCCC(C)CC',
    #          'Oc1cccc(O)c1', 'CCCCCC(C)C', 'Cc1cccc(O)c1', 'CCCC(C)C(=O)O', 'N=Nc1ccccc1', 'ClC1=C(Cl)CC(Cl)C1',
    #          'Cc1ccc(N)cc1', 'O=C1C=CC(=O)C=C1', 'CCCC(C)(O)CO', 'Clc1cncc(Cl)c1', 'CCCCCC(=O)O', 'POc1ccccc1',
    #          'OCc1ccccc1', 'O=C(O)C(F)CCF', 'NCc1ccccc1', 'CCCCCCC=O', 'NC(=O)CCC(=O)O', 'CCCCC(O)CC', 'Clc1ccc(Cl)cc1',
    #          'O=C(O)CCC(=O)O', 'CCN1CCCCC1', 'Nc1ccnc(=O)n1', 'O=CNCC(O)CO', 'OCC(O)C(O)CO', 'CC(C)C(C)C(C)C',
    #          'Cc1ccccc1Cl', '[nH]c(=O)[nH]c=O', 'ncc1ccccc1', 'CCCC(=O)OCC', 'C=C(C)C(=O)OCC', 'CC(NC=O)C(=O)O',
    #          'Nc1ncnc(N)n1', 'CC(O)CNCCN', 'FCCCCCCF', 'Nc1ccccc1S', 'C1CN2CSSC2=N1', 'Cc1ccccc1', 'Clc1ccccc1',
    #          'Oc1ccccc1', 'Nc1ccccc1', 'Brc1ccccc1', 'CCCCC(=O)O', 'O=c1ncncn1', 'CCCC(=O)OC', 'Nc1ccncn1', 'CCCCCCC',
    #          'CCC(O)CCO', 'NC1CCCCC1', 'Nc1ncncn1', 'O=CCCC(=O)O', 'ClC1CCCCC1', 'Clc1cccnc1', 'OCCNCCO', 'OCCC(O)CO',
    #          'CCCC(O)CO', 'CC(C)CC(=O)O', 'CCCCCCF', 'CCCCNCC', 'C=C(C)C(=O)OC', 'Clc1ncncn1', 'CCC(F)C(=O)O',
    #          'ClCCC(Cl)CCl', 'CCO[PH](O)=S', 'CO[PH](=S)OC', 'CC=CC=CCC', 'Fc1ccccn1', 'CCCC(=O)NC', 'Clc1ccccn1',
    #          'Cc1ccccn1', 'CO[PH](=O)OC', 'cccccco', 'Cc1cnccn1', 'COP(=S)(S)OC', 'CCCC(C)(C)O', 'CN1CCCCC1',
    #          'CCC(CCl)CCl', 'CCCNCCC', 'oc1ccccc1', 'CCCC(C)(C)C', 'O=C1C=CC(=O)N1', 'CC(=O)CC(C)C', 'CCCC(C)CCl',
    #          'c1ccccc1', 'CCCC(=O)O', 'c1ccncc1', 'CC(O)CCO', 'c1cncnc1', 'CCCC(C)O', 'CCCC(C)C', 'C1CCCCC1',
    #          'CCC(O)CO', 'CCCCCC', 'c1ncncn1', 'C1CNCCN1', 'CCCC(N)=O', 'CCOCCO', 'CC(C)CC=O', 'CCC(C)CCl', 'c1cnccn1',
    #          'CCCCOP', 'CCC(C)(C)C', 'NC(=O)CCS', 'ccccCC', 'C1COCCN1', 'C=C(C)C(=O)O', 'cccccc', 'O=CCCC=O',
    #          'c1cc[nH]c1', 'CCCCCO', 'c1c[nH]cn1', 'CC(CO)CO', 'CC(C)(C)NO', 'NC(=O)CCO', 'ClCCCCCl', 'O=C(O)CCS',
    #          'CO[PH](O)=S', 'O=C1CC(S)N1', 'CC(O)C(=O)O', 'NC(=O)NC=O', 'CCOC(C)=O', 'CC(=O)CC=O', 'O=C1CCCO1',
    #          'CC(C)C(C)O', 'CCC(=O)NC', 'CCC1(C)CO1', 'CCC(=O)CC', 'Cn1cccn1', 'CN(C)CCO', 'OCCC1CO1', 'CN[SH](=O)=O',
    #          '[nH]c([nH])=O', 'CC=CC(=O)O', 'C=C(C)CCC', 'CC(O)CC=O', 'CN1C=NCC1', 'CCCCO', 'CCC(C)C', 'CCCC=O',
    #          'CCCCC', 'O=[SH](=O)O', 'CCC(C)O', 'CCC(=O)O', 'CCC(C)=O', 'CCC(N)=O', 'CCNCC', 'N[SH](=O)=O', 'C=C(C)C=O',
    #          'CC(O)CO', 'ccccc', 'CCCCCl', 'C=CC(=O)O', 'CC=CCC', 'CC(C)(C)N', 'O=[PH](O)O', 'COPOC', 'ClCCCCl',
    #          'COC(C)=O', 'CC(C)(C)O', 'c1cncn1', 'NCC(N)=O', 'CCOC=O', 'O=CCCS', 'c1cnnc1', 'CCC1CO1', 'COCCO', 'CCCCN',
    #          'CCNC=O', 'C=CCCC', 'CC(C)C=O', 'CCOCC', 'NCC(=O)O', 'c1ccoc1', 'cccc=O', 'COC(N)=O', 'O=C(O)CO',
    #          'C[SH](=O)=O', 'CC(C)CO', 'CCC1CC1', 'OCPCO', 'c1ncnn1', 'CS(=O)(=O)O', 'CNC(C)=O', 'CNC(=O)O', 'c1ccsc1',
    #          'FCCCCl', 'CCN(C)C', 'CCC(C)Br', 'cccc', 'CCCC', 'CCC=O', 'CC(=O)O', 'CC(C)C', 'cncn', 'CC(N)=O', 'CC(C)O',
    #          'OCCO', 'CCNC', 'CC=CC', 'CCCO', 'NCCO', 'CCCCl', 'CC(C)=O', 'NC(N)=O', 'ccc=O', 'CCCN', 'CC(C)N',
    #          '[nH]cn', '[nH]c=O', 'NCC=O', 'CC1CO1', 'nc[nH]', 'NC(=O)O', 'ccnc', 'COC=O', 'CNC=O', 'O=CCCl', 'O=CCO',
    #          'COP=O', 'OC[PH]', 'nccn', 'CCC#N', 'CN(C)C', 'CCOP', 'CCOC', 'ncc=O', 'NCCN', 'FCCS', 'C=C(C)C', 'N#CCCl',
    #          'CCC', 'CCO', 'CCN', 'O=CO', 'NC=O', 'O=S=O', 'CC=O', 'ccn', 'CNC', 'ccc', 'c[nH]', 'COP', 'O=[N+][O-]',
    #          'CCCl', 'O=PO', 'C[N+]C', 'C=CC', 'N=CN', 'NC=S', 'OP=S', 'ncn', 'CC[N+]', 'CCS', 'CC=N', 'CC', 'cc', 'CO',
    #          'CN', 'cn', 'O=S', 'OP', 'C[N+]', '[N+]=O', 'CS', 'C=O', 'CCl', 'CBr', 'nn', 'N=N', 'SS', 'C[n+]', 'C#N',
    #          'co', 'cs', 'CF', 'C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I']

    # Mutagenicity 100
    smis_0 = ['O=C1c2ccccc2C(=O)c2c(O)cccc21', 'O=C1c2ccccc2C(=O)c2ccccc21', 'c1cc2ccc3cccc4ccc(c1)c2c34',
             'c1ccc2nc3ccccc3cc2c1', 'c1ccc2c(c1)ccc1ccccc12', 'c1ccc2cc3ccccc3cc2c1', 'cccc1cccc2ccccc12',
             'ccccccc1ccccc1', 'O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'c1ccc(c2ccccc2)cc1', 'Nc1ccc2ccccc2c1',
             'Nc1cccc2ccccc12', 'c1ccc2ccccc2c1', 'c1ccc2ncccc2c1', 'ccccc1ccccc1', 'Cc1ccc([N+](=O)[O-])cc1',
             'Cc1ccccc1[N+](=O)[O-]', 'Nc1ccc([N+](=O)[O-])cc1', 'c1ccc2occcc2c1', 'O=[N+]([O-])c1ccccc1',
             'cccc1ccccc1', 'c1ccc(C2CO2)cc1', 'Cc1ccc([N+](=O)[O-])o1', 'cccccccc', 'ccc1ccccc1', 'O=Cc1ccccc1',
             'CCc1ccccc1', 'Cc1ccccc1N', 'CNc1ccccc1', 'N=Nc1ccccc1', 'O=[N+]([O-])c1cccs1', 'Cc1ccccc1', 'Nc1ccccc1',
             'Oc1ccccc1', 'cccc[N+](=O)[O-]', 'nc1ccccc1', 'c1ccccc1', 'c1ccncc1', 'Cc1ccco1', 'c1cncnc1', 'CCOCCO',
             'ccccC', 'O=[SH](=O)O', 'c1cncn1', 'CC(O)CO', 'ccccn', 'CCNCC', 'CCNC=O', 'Cnc(n)N', 'cccc', 'CCNC',
             'CC1CO1', 'CC(N)=O', 'cccn', 'OCCO', 'CC(=O)O', 'CCCC', 'CCCO', 'Cncn', 'cncn', 'CC(C)O', 'O=CCCl', 'CCO',
             'O=[N+][O-]', 'CCN', 'CC=O', 'ccn', 'CCCl', 'O=S=O', 'CCC', 'CNC', 'NC=O', 'ncn', 'ccc', 'O=CO', 'ClCCl',
             'C1CO1', 'cc', 'CC', '[N+][O-]', 'cn', 'CO', 'CN', 'O=S', 'CCl', 'N=O', 'NO', 'H', 'B', 'C', 'N', 'O', 'F',
             'Na', 'P', 'S', 'Cl', 'Ca', 'Br', 'I']
    smis_1 = ['CCCCCCCCCCCC', 'OCC1OCC(O)CC1O', 'c1ccc2[nH]ccc2c1', 'c1ccc2ncccc2c1', 'CC(=O)Nc1ccccc1',
             'c1ccc2ccccc2c1', 'O=C(O)c1ccccc1', 'FC(F)c1ccccc1', 'CCCCCCCCC', 'CCCCCCCC', 'O=Cc1ccccc1',
             'Clc1cccc(Cl)c1', 'ccc1ccccc1', 'FCc1ccccc1', 'COc1ccccc1', 'Cc1ccc(O)cc1', 'Oc1ccccc1Br', 'CCc1ccccc1',
             'Cc1ccccc1', 'Clc1ccccc1', 'Oc1ccccc1', 'Nc1ccccc1', 'Brc1ccccc1', 'CCCCC(=O)O', 'c1ccccc1', 'c1ccncc1',
             'CCCC(=O)O', 'c1cncnc1', 'CCOCCO', 'CC(O)CCO', 'CCCC(C)C', 'CCCC(C)O', 'CCC(O)CO', 'C=C(C)C(=O)O',
             'C1CCCCC1', 'CCCCO', 'CCC(C)C', 'CCCC=O', 'CCCCC', 'O=[SH](=O)O', 'CCC(C)O', 'CCC(=O)O', 'CCNCC',
             'CCC(C)=O', 'C=CC(=O)O', 'ccccc', 'COPOC', 'cccc', 'CCCC', 'CC(=O)O', 'cncn', 'CC(C)C', 'CC(C)O',
             'CC(N)=O', 'CCCO', 'OCCO', 'CCNC', 'CCC=O', 'NCCO', 'CC=CC', 'CCCCl', 'ccc=O', 'CCO', 'CCC', 'CCN', 'O=CO',
             'NC=O', 'O=S=O', 'CC=O', 'CNC', 'ccn', 'ccc', 'COP', 'c[nH]', 'CCF', 'O=[N+][O-]', 'CC', 'cc', 'CO', 'CN',
             'cn', 'O=S', 'OP', 'C=O', 'CS', '[N+][O-]', 'C[N+]', 'H', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl',
             'Ca', 'Br', 'I']

    # BBBP 500
    # smis_0 = ['CC(C)C1NC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)[C@@H](C(C)C)NC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)[C@@H](C(C)C)NC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC1=O', 'CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)[C@@H](C(C)C)NC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)[C@@H](C(C)C)NC(=O)[C@@H](C(C)C)OC1=O', 'CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)[C@@H](C(C)C)NC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC1=O', 'CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC(=O)[C@H](C(C)C)NC(=O)[C@H](C)OC(=O)CNC(=O)[C@@H](C(C)C)OC1=O', 'CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)CC(C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O', 'CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(ccc4c3C2=O)NC(=O)/C(C)=C\\C=C\\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C', 'CO[C@H]1/C=C/O[C@@]2(C)Oc3ccc4cc(c5scnc5c4c3C2=O)NC(=O)/C(C)=C\\C=C\\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C', 'CO[C@H]1/C=C/O[C@@]2(C)Oc3cc(O)c4c(O)c(ccc4c3C2=O)NC(=O)/C(C)=C\\C=C\\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C', 'CC(C)[C@H](NC(=O)[C@H](C)O)C(=O)O[C@@H](C(=O)NCC(=O)O[C@@H](C)C(=O)N[C@H](C(=O)O[C@@H](C(=O)NCC=O)C(C)C)C(C)C)C(C)C', 'CC[C@H](CC(O)[C@@H](OC(=O)c1ccccc1)[C@H]1C(C)[C@@H](O)C[C@H]2OC[C@@]12OC(C)=O)OC(=O)[C@H](O)[C@@H](N)c1ccccc1', 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1', 'CC[C@H](CC(O)[C@H](C[C@]1(OC(C)=O)COC1)OC(=O)c1ccccc1)OC(=O)[C@H](O)[C@@H](N)c1ccccc1', 'CC[C@H](CC(O)[C@H](CC(CO)OC(C)=O)OC(=O)c1ccccc1)OC(=O)[C@H](O)[C@@H](N)c1ccccc1', 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1', 'CC[C@H](O[C@H]1C[C@@](C)(OC)[C@@H](O)[C@H](C)O1)[C@@H](C)C(=O)O[C@H](CC)[C@@](C)(O)[C@H](O)[C@@H](C)C=O', 'CO[C@@H](/C=C/O)[C@@H](C)[C@@H](OC(C)=O)[C@H](C)[C@H](O)[C@H](C)[C@@H](O)[C@@H](C)/C=C/C=C(/C)C(N)=O', 'CC1(C)S[C@@H]2[C@H](NC(=O)C(C(=O)Oc3ccc4c(c3)CCC4)c3ccccc3)C(=O)N2[C@H]1C(=O)O', 'C[C@@H]1[C@@H](O)[C@@H](C)C[C@]2(CO2)C(=O)[C@H](C)[C@@H](O)[C@@H](C)[C@@H](C)OC(=O)[C@H](C)[C@H]1O', 'CN(C)[C@@H]1C(=O)/C(=C(\\N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4[C@@](C)(O)[C@H]3C[C@@H]12', 'CN(C)[C@@H]1C(=O)/C(=C(\\N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4[C@@H](O)[C@H]3C[C@@H]12', 'CC1(C)S[C@@H]2[C@H](NC(=O)C(C(=O)Oc3ccccc3)c3ccccc3)C(=O)N2[C@H]1C(=O)O', 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)C[C@H]3OC1CC(N)CC(C)O1', 'CN(C)[C@@H]1C(=O)/C(=C(\\N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4C[C@H]3[C@H](O)[C@@H]12', 'CN(C)[C@@H]1C(=O)/C(=C/O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4[C@@](C)(O)[C@H]3C[C@@H]12', 'C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@H](C)[C@H](/C=C/O)OC', 'CN(C)[C@@H]1C(=O)/C(=C(\\N)O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4C[C@H]3C[C@@H]12', 'NCC1OC(OC2C(N)CC(N)C(OC3OC(CO)C(O)C(N)C3O)C2O)C(N)C(O)C1O', 'CC[C@H](O[C@H]1C[C@@](C)(OC)[C@@H](O)[C@H](C)O1)[C@@H](C)C(=O)O[C@H](CC)C(C)O', 'CN(C)C1C(=O)/C(=C(\\N)O)C(=O)C2(O)C(=O)C3=C(O)c4c(O)cccc4C(C)(O)C3CC12', 'CC[C@H](CC(O)[C@H](CC(CO)OC(C)=O)OC(=O)c1ccccc1)OC(=O)CO', 'CC(=O)N1CCN(C(=O)Cc2ccc(C(F)(F)F)cc2)[C@@H](CN2CC[C@@H](O)C2)C1', 'CO[C@@]1(NC(=O)C2SCS2)C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)CS[C@@H]21', 'CC(=O)N1CCN(C(=O)Cc2ccc([N+](=O)[O-])cc2)[C@@H](CN2CC[C@H](O)C2)C1', 'CC(=O)N1CCN(C(=O)Cc2cccc([N+](=O)[O-])c2)[C@@H](CN2CC[C@H](O)C2)C1', 'CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CS)CS[C@H]12)c1csc(N)n1', 'CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O', 'CN[C@@H]1[C@H](O)[C@H](NC)[C@H]2O[C@@]3(O)C(=O)CCO[C@H]3O[C@@H]2[C@H]1O', 'CC(=O)N1CCN(C(=O)Cc2ccc(Cl)c(Cl)c2)[C@@H](CN2CC[C@@H](O)C2)C1', 'CSc1ccc(CC(=O)N2CCN(C(C)=O)C[C@@H]2CN2CC[C@@H](O)C2)cc1', 'CC(=O)N1CCN(C(=O)Cc2cc(F)cc(F)c2)[C@@H](CN2CC[C@@H](O)C2)C1', 'COc1cccc(CC(=O)N2CCN(C(C)=O)C[C@@H]2CN2CC[C@@H](O)C2)c1', 'COc1ccc(CC(=O)N2CCN(C(C)=O)C[C@@H]2CN2CC[C@@H](O)C2)cc1', 'CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O', 'CN(C)[C@H]1C[C@](O)(C(=O)C=C(O)c2ccccc2O)C(=O)/C(=C(/N)O)C1=O', 'CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccc(O)cc3)[C@H]2SC1', 'CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(C)CS[C@H]12)c1csc(N)n1', 'CC(=O)N1CCN(C(=O)Cc2cccc(F)c2)[C@@H](CN2CC[C@@H](O)C2)C1', 'CCO[C@@H](CC(O)CC)c1ccc2c(c1O)C(=O)c1c(O)cccc1C2=O', 'CO[C@@H]1CO[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](N)C[C@@H]2N)[C@H](N)C1', 'NC[C@@H]1CC[C@@H](N)[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](N)C[C@@H]2N)O1', 'NC(C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CS)CS[C@H]12)c1ccc(O)cc1', 'CO[C@@]1(NC(=O)CS)C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)CS[C@@H]21', 'Cn1nnnc1SCC1=C(C(=O)O)N2C(=O)C(NC(=O)CNC=O)[C@H]2SC1', 'CN(C)C[C@@H]1C[C@H]2C(=C(O)c3c(O)cccc3[C@@]2(C)O)C(=O)[C@]1(O)C=O', 'CN(C(=O)Cc1ccccc1N)[C@@H](CN1CC[C@@H](O)C1)c1ccccc1', 'CC(=O)N1CCN(C(=O)Cc2ccccc2)[C@@H](CN2CC[C@@H](O)C2)C1', 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)C[C@@H]3O', 'CC(=O)N1CCN(C(=O)Cc2ccccc2)[C@@H](CN2CC[C@H](O)C2)C1', 'CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccccc3)[C@H]2SC1', 'CO[C@@]1(NC(C)=O)C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)CS[C@@H]21', 'Cc1nnc(SCC2=C(C(=O)O)N3C(=O)[C@@H](NC(=O)CO)[C@H]3SC2)s1', 'Cn1nnnc1SCC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)CO)[C@H]2SC1', 'Nc1nc(CC(=O)N[C@@H]2C(=O)N3C(C(=O)O)=C(CS)CS[C@H]23)cs1', 'CC(C)[C@H](NC(=O)[C@H](C)O)C(=O)O[C@@H](C(=O)NCC=O)C(C)C', 'CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O', 'CCO[C@@H](CCO)c1ccc2c(c1O)C(=O)c1c(O)cccc1C2=O', 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)CC3O', 'Nc1nc(CC(=O)N[C@@H]2C(=O)N3C(C(=O)O)=C(CO)CS[C@H]23)cs1', 'C=CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)Cc3csc(N)n3)[C@H]2SC1', 'CN1CCN(c2c(F)cc3c(=O)c(C(=O)O)cn(CCF)c3c2F)CC1', 'CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)Cc3csc(N)n3)[C@H]2SC1', 'NCCO[C@@H](CN)O[C@H]1[C@H](O)[C@@H](O)[C@H](N)C[C@@H]1N', 'Nc1nc(CC(=O)N[C@@H]2C(=O)N3C(C(=O)O)=CCS[C@H]23)cs1', 'CCOC(=O)[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@@H](C)CC1CO1', 'CCNC[C@H]1CN(C(C)=O)CCN1C(=O)Cc1ccccc1', 'CC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(COC(C)=O)CS[C@H]12', 'COC1C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)CS[C@H]12', 'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O', 'CCC[C@@](C)(O)CO[C@@H]1O[C@H](C)C[C@H](N(C)C)[C@H]1O', 'CC[C@H](CC(O)[C@@H](O)CC(CO)OC(C)=O)OC(=O)CO', 'CO/N=C(\\C(=O)NC[C@H](C)NOCC(=O)O)c1csc(N)n1', 'Cc1cn([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O', 'N/C(O)=C(\\C=O)C(=O)C(O)C(=O)C=C(O)c1ccccc1O', 'CC(C)[C@@H](OC(=O)CNC(=O)[C@H](C)O)C(=O)NCC=O', 'NC1CC(N)C(OC2OC(CO)C(O)C(N)C2O)C(O)C1O', 'Cn1nnnc1SCC1=C(C=O)N2C(=O)[C@@H](N)[C@H]2SC1', 'CON=CC(=O)N[C@@H]1C(=O)N2C(C=O)=C(CO)CS[C@H]12', 'CNC1C(O)C(OC2C(N)CC(N)C(O)C2O)OCC1(C)O', 'O=C/C(=C/O)C(=O)C(O)C(=O)C=C(O)c1ccccc1O', 'CC1(C)S[C@@H]2[C@H](NC(=O)CN)C(=O)N2[C@H]1C(=O)O', 'CO[C@@H](/C=C/O)[C@@H](C)[C@@H](OC(C)=O)[C@H](C)CO', 'NCC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CS)CS[C@H]12', 'CC(=O)N1CCN(C(C)=O)[C@@H](CN2CC[C@@H](O)C2)C1', 'CC1(C)S[C@@H]2[C@H](NC(=O)CO)C(=O)N2[C@H]1C(=O)O', 'C[C@H]([C@@H](O)[C@@H](C)CC1CO1)[C@H](O)[C@@H](C)C=O', 'CC1(C)S[C@@H]2[C@H](NC(=O)CN)C(=O)N2[C@H]1C(=O)[O-]', 'C[C@@H]1OCC[C@@H]2O[C@H]3CC(=O)CO[C@H]3O[C@H]12', 'CN(CCN1CC[C@@H](O)C1)C(=O)Cc1ccccc1N', 'CC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CS)CS[C@H]12', 'CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)CN)[C@H]2SC1', 'COc1cccc2c1C(=O)c1c(O)ccc(O)c1C2=O', 'CC(=O)N[C@@H]1C(=O)N2[C@@H]1SC(C)(C)[C@@H]2C(=O)O', 'CC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(CO)CS[C@H]12', 'Cc1nnc(SCC2=C(C(=O)O)N3C(=O)CC3SC2)s1', 'CC(=O)OCC1=C(C(=O)O)N2C(=O)[C@@H](N)[C@H]2SC1', 'C[C@@H](CNC(=O)Cc1csc(N)n1)NOCC(=O)O', 'C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(S)[C@H](C)[C@H]12', 'CNC[C@H]1O[C@@]2(O)C(=O)CCO[C@H]2O[C@@H]1CO', 'CC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(C)CS[C@H]12', 'CC1(C)S[C@@H]2[C@H](NC=O)C(=O)N2[C@H]1C(=O)O', 'NCCO[C@H]1[C@H](O)[C@@H](O)[C@H](N)C[C@@H]1N', 'NCC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=CCS[C@H]12', 'CC(=O)N[C@@H]1C(=O)N2C(C=O)=C(CO)CS[C@H]12', 'CCCO[C@@H]1O[C@H](C)C[C@H](N(C)C)[C@H]1O', 'CC[C@@H](O)CC(O)[C@@H](O)CC(CO)OC(C)=O', 'CC(O)CCO[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1', 'O=C(O)c1cn(CCF)c2c(F)cc(F)cc2c1=O', 'O=C1c2ccccc2C(=O)c2c(O)ccc(O)c21', 'O=C1c2cccc(O)c2C(=O)c2c(O)cccc21', 'CC(=O)N[C@@H]1C(=O)N2C(C(=O)O)=CCS[C@H]12', 'CC[C@@H](OC(=O)[C@H](C)[C@@H](O)CC)C(C)O', 'C[C@H](O)C(=O)NCC(=O)OCC(=O)NCC=O', 'CCn1cc(C(=O)O)c(=O)c2cc(F)cc(F)c21', 'CC(N)C(CO)OC1OC(CO)C(O)C(N)C1O', 'O=CC(C(=O)Oc1ccccc1)c1ccccc1', 'O=C1c2cccc(O)c2C(=O)c2cccc(O)c21', 'CC(=O)NC1C(=O)N2C(C(=O)O)=C(C)CS[C@H]12', 'CC1(C)SC2C(NC(=O)CN)C(=O)N2C1C(=O)O', 'CC1Oc2ccc3ccc4scnc4c3c2C1=O', 'O=C1c2ccccc2C(=O)c2c(O)cccc21', 'CCNC[C@H]1CN(C(C)=O)CCN1C(C)=O', 'CC1(C)S[C@@H]2[C@H](N)C(=O)N2[C@H]1C(=O)O', 'CC1Oc2cc(O)c3c(O)cccc3c2C1=O', 'CC(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O', 'CC(=O)OCC1=C(C(=O)[O-])N[C@H](CN)SC1', 'CC1=C[C@@H](C=O)[C@]2(O)CCO[C@@H]2[C@@H]1O', 'CO[C@H]1CC(OCCO)O[C@@H](C)[C@@H]1O', 'COC1C(=O)N2C(C(=O)O)=C(CS)CO[C@H]12', 'C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=CS[C@H]12', 'CC1(C)[C@H](C(=O)O)N2C(=O)C[C@H]2S1(=O)=O', 'COC1C(=O)N2[C@@H]1SC(C)(C)[C@@H]2C(=O)O', 'CC(=O)O[C@H]1CO[C@H](C)C[C@@H]1N(C)C', 'O=CC(O)C(=O)C=C(O)c1ccccc1O', 'O=CCNC(=O)COC(=O)CNC(=O)CO', 'N[C@@H]1C[C@H](N)[C@@H](O)[C@H](O)[C@H]1O', 'N[C@@H]1C(=O)N2C(C=O)=C(CO)CS[C@H]12', 'C[C@H]1OC[C@H](O)[C@@H](N(C)C)[C@@H]1O', 'CC1(C)[C@H](C(=O)O)N2C(=O)C[C@H]2S1=O', 'CCO[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1', 'O=CNCC(=O)NC1CN(CC(=O)O)C1=O', 'CC(=O)NC1C(=O)N2C(C(=O)O)=CCCC12', 'CC[C@H](O)[C@@H](C)[C@H](O)[C@@H](C)C=O', 'C[C@@H]1OCC[C@@H]2O[C@@H](C)CO[C@H]12', 'O=C1CC[C@@H](C(=O)Oc2ccccc2)N1', 'NCC(=O)N[C@H]1CN(CC(=O)O)C1=O', 'CC1(C)S[C@@H]2CC(=O)N2[C@H]1C(=O)O', 'N[C@H]1[C@H](O)[C@@H](CO)OC[C@@H]1O', 'CCn1ccc(=O)c2cc(F)cc(F)c21', 'NC[C@H]1OC[C@H](O)[C@@H](O)[C@@H]1O', 'NC12C[C@H]3C[C@@H](C1)CC(C=O)(C3)C2', 'N[C@H]1[C@H](O)[C@@H](O)CO[C@@H]1CO', 'NCCOC1OC(CN)C(O)C(O)C1N', 'CC(O)CCOC1CC(N)C(O)C(C)O1', 'O=C(O)CN1C[C@H](NC(=O)CO)C1=O', 'OC[C@H]1OC[C@H](O)[C@@H](O)[C@@H]1O', 'C[C@H](N)CNC(=O)Cc1csc(N)n1', 'CC(=O)O[C@H]1[C@H](C)OCC[C@@]1(C)O', 'OC[C@H]1OC[C@@H](O)[C@@H](O)[C@@H]1O', 'NCCN[C@H]1CN(CC(=O)O)C1=O', 'CC(=O)N[C@H]1CN(CC(=O)O)C1=O', 'C[C@@H]1C[C@H](N(C)C)[C@@H](O)CO1', 'NCC(=O)NC1CN(CC(=O)O)C1=O', 'CN1CC(=O)c2ccccc2S1(=O)=O', 'C[C@@H](O)[C@@H]1CN(CC(=O)O)C1=O', 'C[C@@H]1OC[C@H](O)[C@H](O)[C@H]1O', 'CC(O)[C@@H](O)CC(CO)OC(C)=O', 'CC[C@H]1OC(CCO)CC[C@@H]1C', 'CC(O)CCOC1CC(N)CC(C)O1', 'C[C@H]1C(S)=C(C(=O)O)N2C(=O)CC12', 'CC1Oc2cc(O)cc(C=O)c2C1=O', 'C[C@@H]1OCC[C@H](N(C)C)[C@@H]1O', 'O[C@@H]1[C@@H](O)[C@H](O)OC[C@H]1O', 'CC(=O)N(C)CCN1CC[C@@H](O)C1', 'CC[C@@H](OC(C)=O)[C@H](C)CO', 'CO[C@]1(C)CCO[C@@H](C)[C@@H]1O', 'CC(=O)NC1CN(CC(=O)O)C1=O', 'CO[C@H]1CCO[C@@H](C)[C@@H]1O', 'CC1(C)SC2CC(=O)N2C1C(=O)O', 'O=c1ccnc2c(F)cc(F)cc12', 'C[C@@H]1NC(=O)[C@H]1NC(=O)C=N', 'N=CC(=O)N[C@@H]1C(=O)NC1CO', 'Cc1cc(-c2ccccc2Cl)no1', 'O=C(O)C1=C(S)C[C@@H]2CC(=O)N12', 'CCOC1CCC(N(C)C)C(C)O1', 'OC[C@H]1OCC[C@@H](O)[C@@H]1O', 'CC1(C)NC(=O)[C@H]1NC(=O)C=N', 'O=C1CCO[C@@H](OCCO)C1O', 'O=CC=C(O)c1ccccc1O', 'CC(=O)N1CCN(C(C)=O)CC1', 'CC1Oc2cc(O)ccc2C1=O', 'C[C@@H]1OCC[C@H](N)[C@@H]1O', 'O=c1ccnc2ccc(F)cc12', 'OC[C@H]1OC[C@@H](O)[C@@H]1O', 'O=CCC(=O)Oc1ccccc1', 'c1ccc(C2=NCCCN2)cc1', 'CC[C@@H](O)[C@H]1OCCC1O', 'CC(C)[C@H]1OCCC[C@@H]1C', 'CCC[C@H](C)C[C@@](C)(O)CO', 'C[C@H]1OCC[C@@](C)(O)[C@@H]1O', 'O=c1ccoc2cc(O)ccc12', 'CC1OCC(O)C(N(C)C)C1O', '[N-]=[N+]=N[C@H]1CCO[C@@H]1CO', 'N[C@H]1CN(CC(=O)O)C1=O', 'CC(=O)NCCNCC(=O)O', 'NC1C(O)COC(CO)C1O', 'CC1CC(N(C)C)C(O)CO1', 'NC1CC(N)C(O)C(O)C1O', 'Cn1nnnc1SCCCS', 'C/C=C/[C@H](C)[C@H](O)CC', 'COC1(C)CCOC(C)C1O', 'COC1CN(CC(=O)O)C1=O', 'NCC1OCC(O)C(O)C1O', 'CCC(=O)Oc1ccccc1', 'CCOC(C)OCCC(C)O', 'O=c1ccnc2ccccc12', 'C[C@@H]1CC(N)C[C@H](C)C1', 'c1c[nH]c2cccnc2c1', 'NCCOCC(N)C(O)CO', 'Cc1nnc(SCCCS)s1', 'OC1[C@@H](O)COC[C@@H]1O', 'CO[C@H]1CCO[C@@H](C)C1', 'C[C@H]1CN[C@@H](CO)OC1', 'COC1C(O)COC(C)C1O', 'CNC1C(O)COCC1(C)O', 'CSC1OCC(O)C(O)C1O', 'CC(C)SCCNC(=O)CN', 'NC(=O)CN1CC(O)CC1=O', 'C[C@@H]1C[C@@H](O)CC(=O)O1', 'CC(O)CCCCN(C)C', 'CC[C@H](O)[C@@H](C)C=O', 'N[C@H]1CN(CC=O)C1=O', 'NCC1=CC[C@@H](N)CO1', 'COc1ccccc1C=O', 'O=C(O)c1ccccc1O', 'c1ccc2[nH]ccc2c1', 'NC1CN(CC(=O)[O-])C1=O', 'O[C@@H]1COCC[C@@H]1O', 'cccc1cccc[nH]1', 'CCC[C@@H]1CCN(C)C1', 'CC(=O)OC(CO)CCO', 'c1ccc2nccnc2c1', 'cccc1cC(=O)C(C)O1', 'C[C@H]1C[C@@H](O)CCO1', 'CC1OCCC(C)(O)C1O', 'O[C@@H]1COC[C@@H](O)C1', '[nH]c1ccc(Cl)c(Cl)c1', 'Cc1cnc(=O)[nH]c1=O', 'O=C(O)CN1CCC1=O', 'O=Cc1cccc(O)c1', 'CC(=O)N1CCNCC1', 'O=Cc1ccccc1O', 'O=S(=O)c1ccccc1', 'C[C@@H](N)[C@H](O)CO', 'CC1OCCC(O)C1O', 'OC1COCC(O)C1O', 'OCNc1ccccn1', 'CC1OCCC(N)C1O', 'NCC1CCC(N)CO1', 'CCOC[C@H](O)CC', 'COCCOCCCS', 'CC(=O)NC[C@H](C)N', 'CN1CCN(C=N)CC1', 'CCCC1CCN(C)C1', '[N-]=[N+]=N[C@H]1CCOC1', 'CCN1CC[C@@H](O)C1', 'O=Cc1ccccc1', 'O=CCNC(=O)CO', 'Oc1ccccc1O', 'CCOC(C)OCC', 'Clc1ccccc1Cl', 'OC[C@@H]1CCCN1', 'CCC[C@@](C)(O)CO', 'NCc1ccccc1', 'CC[C@@H](O)C(C)O', 'CN1CCCCCC1', 'N=CC(=O)NCC=O', 'COc1ccccc1', 'C[C@H]1CNCCN1', 'NCC[C@H](O)C=O', 'CC1OCCCC1=O', 'CC[C@@H](N)C(=O)O', 'NCCNCC(=O)[O-]', 'COC(=O)C(C)(C)C', 'N[C@@H]1CCCOC1', 'O=c1ccncc1Cl', 'Cc1oc(=O)oc1C', 'CC(=O)OCCCS', 'CC1OCC(O)C1O', 'CCOCOCCO', 'CCC(O)C(C)CO', 'ccc1scnc1c', 'CC(=O)NCC(N)=O', '[N+]=N[C@H]1CCOC1', 'Oc1ccccc1', 'O=CNCC(=O)O', 'CC(=O)NCCN', 'CCCCN(C)C', 'Clc1ccccc1', 'OCCOCCO', 'CCCOC(C)=O', 'NCCOCCN', 'CCNC(=O)CN', 'CN1CCNCC1', 'NCCNCC=O', 'CC(N)C(O)CO', 'COC(C)CCO', 'Fc1ccccc1', 'Cc1ccccc1', 'CC(=O)NCC=O', 'O=c1ccncc1', 'NCCOCCO', 'CCOCCCO', 'C[C@@H](C=O)CO', 'CC[C@@H](O)CC', 'NC(=O)CC(=O)O', 'CCOC(=O)CC', 'CC(=O)OCCO', 'N[C@H]1CCOC1', 'CC1CNCCN1', 'CCNC(=O)CC', 'CO[C@@H](C)CO', 'O=Cc1ccco1', 'NC(N)=NCCO', 'C[C@H]1CCCO1', 'NC1CCCOC1', 'O=CNCC(=O)[O-]', 'O=C1C[C@@H](S)N1', 'Nc1ccccc1', 'c1ccccc1', 'CCNCCN', 'Nc1nccs1', 'c1ccncc1', 'CC(O)CCO', 'CCOCCO', 'Cn1cnnn1', 'CN(C)CCO', 'CCCC(C)C', 'COC/C=C/O', 'cccc[nH]', 'CCNC(C)=O', 'CC[C@H](C)O', 'C1CNCCN1', 'Cc1nncs1', 'C1=CCC=CC1', 'OCC(O)CO', 'CCCCCO', 'COCC(C)O', 'O=CCNC=O', 'C[C@@H](O)CO', 'CC(O)CCN', 'c1cn[nH]n1', 'Cc1cscn1', 'c1nc[nH]n1', 'Cc1ccno1', 'CCOC(C)=O', 'cc(O)ccO', 'CC(C)CCO', 'CC(C)(O)C=O', 'C/C=C/CCC', 'C1CCOCC1', 'CC1N=CCN1', 'CN1CCSC1', 'Cccnc=O', 'NCC(=O)O', 'c1cscn1', 'O=CC=CO', 'CCN(C)C', 'COCCO', 'CNCCO', 'O=C1CCN1', 'CCOCC', 'SCCCS', 'c1nnnn1', 'CC(O)C=O', 'CC(O)CO', 'c1ccoc1', 'c1nncs1', 'c1ccsc1', 'ccccO', 'CCNCC', 'CCC(=O)O', 'C/C=C/CC', 'CCC(N)=O', 'O=C(O)CO', 'OCCCS', 'CNCC=O', 'NC(=O)CO', 'ccccc', 'nccc=O', 'COC(C)=O', 'CCCCC', 'CNC(C)=O', 'c1cnoc1', 'FC(F)(F)S', 'CC(N)C=O', 'ccc(c)C', 'CC(Cl)CN', 'CCNC=N', 'O=ccco', 'CC=CCC', 'C1CCCC1', 'c1cnnn1', 'cccc', 'NCCO', 'CCCO', 'CCNC', 'OCCO', 'CCCS', 'CC(=O)O', 'CC(C)S', 'O=CCO', 'CCC=O', 'CCOC', 'NCC=O', 'CC(N)=O', 'CCCC', 'CC(C)C', 'CC(C)O', 'CC1CO1', 'CCCN', 'O=S(=O)O', 'CC(C)N', 'FC(F)S', 'COC=O', 'NS(=O)=O', 'C/C=C/C', '[nH]c=O', 'O=[NH+][O-]', 'O=S(=O)[O-]', 'nccn', 'O=c(o)o', 'CccC', 'cn[nH]', 'cc(C)o', 'O=cc=O', 'CC=NN', 'FC(F)F', 'CCO', 'CCN', 'CCC', 'CC=O', 'ncs', 'CON', 'O=S=O', 'NC=O', 'O=CO', 'nnn', 'ccC', 'cc=O', 'ccn', 'CCS', 'FCS', 'NCN', 'NCO', 'COC', 'O=[N+][O-]', 'nc=O', 'FCF', '[NH3+][O-]', 'O=co', 'C=CC', 'CC=N', 'CS=O', 'CC', 'cc', 'C=O', 'CO', 'cn', 'O=S', 'nn', 'CN', 'CS', 'co', 'c=O', 'CF', '[N+][O-]', 'NO', 'H', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl', 'Ca', 'Br', 'I']
    # smis_1 = ['CC1(C)OC2[C@@H](C[C@H]3[C@@H]4CCC5=CC(=O)C=C[C@]5(C)[C@@]4(F)[C@@H](O)C[C@]23C)O1', 'CC[C@H](NC(=O)c1c(O)c(-c2ccccc2)nc2ccccc12)c1ccccc1', 'CC[C@H](NC(=O)c1cc(-c2ccccc2)nc2ccccc12)c1ccccc1', 'Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2', 'C[C@]1(C)C[C@@H]2CCC3=CC(=O)C=C[C@]3(C)[C@@]2(F)[C@@H](O)C1', 'C[C@]1(C)C[C@@H]2CCC3=CC(=O)C=C[C@]3(C)[C@H]2[C@@H](O)C1', 'C[C@@]1(CO)C[C@@H]2CCC3=CC(=O)C=C[C@]3(C)C2[C@@H](O)C1', 'C[C@]1(C)C[C@@H]2CCC3=CC(=O)C=C[C@]3(C)C2[C@@H](O)C1', 'CCN(CCN1CCCC1)C(=O)Cc1ccc(Cl)c(Cl)c1', 'CC(=O)N1c2ccccc2Sc2ccc(C(F)(F)F)cc21', 'O=C1CN=C(c2ccccc2)c2cc([N+](=O)[O-])ccc2N1', 'Clc1ccc2c(c1)C(c1ccccc1)=NCc1nncn1-2', 'CC1C=CC(=O)C=C1CCCCC[C@@H]1COC(C)(C)O1', 'CCN1c2ccccc2Sc2ccc(C(F)(F)F)cc21', 'CN(C)CCCN1c2ccccc2Sc2ccccc21', 'CC1C[C@@H]2CCC3=CC(=O)C=C[C@]3(C)C2[C@@H](O)C1', 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21', 'CN1CCC23c4c5ccc(O)c4OC2CCCC3C1C5', 'CN1CCc2cc(Cl)ccc2[C@@H](c2ccccc2)C1', 'CC1CC(O)C2C(C1)C[C@H](F)C1=CC(=O)C=CC21C', 'O=CN1CCN(C(=O)Cc2ccc(Cl)c(Cl)c2)CC1', 'O=C(CCCN1CCC(O)CC1)c1ccc(F)cc1', 'CC1CC(O)C2C(CCC3=CC(=O)C=CC32C)C1C', 'CCN1c2ccccc2Sc2ccc(S(=O)=O)cc21', 'O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1', 'C/C(=C(\\S)CCO)N(C=O)Cc1cnc(C)nc1N', 'O=C(CCCN1CCCCC1)c1ccc(F)cc1', 'O=C(CCCN1CCNCC1)c1ccc(F)cc1', 'NCCCOc1cccc(CN2CCCCC2)c1', 'CC(C)C[C@H]1C[C@@]2(C)C=CC(=O)C=C2CC1', 'CCN1c2ccccc2CCc2ccccc21', 'CCN1c2ccccc2Sc2ccc(Cl)cc21', 'Oc1ccc2c3c1O[C@H]1CCCC(CC2)C31', 'FC(F)(F)c1ccc(Sc2ccccc2)cc1', 'CC(C)C[C@@H]1CCC2=CC(=O)C=C[C@]2(C)C1', 'Nc1nc(=O)c2ncn(COCCO)c2[nH]1', 'CC[C@@H]1CC[C@](O)(C(=O)COC(C)=O)C1C', 'CC(C)C[C@H]1CC2C=CC(=O)C=C2CC1', 'CCN1c2ccccc2Sc2ccccc21', 'CC(C)C[C@@H]1CCC2=CC(=O)C=CC2C1', 'c1ccc(-c2ccc3ccccc3n2)cc1', 'CCOc1cccc(CN2CCCCC2)c1', 'CCCCN1CCN(c2ncccn2)CC1', 'O=[N+]([O-])c1cccc(Cc2ccccc2)c1', 'COc1cccc2c1O[C@@H]([C@H](C)O)C2', 'CN1c2ccccc2CCc2ccccc21', 'CCCC(c1ccccc1)c1ccccc1', 'O=C(c1cccc(Cl)c1)c1ccccc1Cl', 'Fc1ccccc1Cc1cccc(Cl)c1', 'c1ccc2c(c1)CCc1ccccc1C2', 'Clc1ccc2c(c1)Cc1ccccc1S2', 'Clc1cccc(Cc2ccccc2Cl)c1', 'O=C(O)c1cnc2ccc(F)cc2c1=O', 'CN1c2ccccc2Sc2ccccc21', 'CCC1=CC(=O)C=C[C@]1(C)C[C@H](C)O', 'c1ccc2c(c1)COc1ccccc1C2', 'C[C@H](O)C[C@@]1(C)C=CC(=O)C=C1CF', 'CCNCCCC(=O)c1ccc(F)cc1', 'c1ccc2c(c1)CCc1ccccc1N2', 'CCN1CC(=O)Nc2ccc(Cl)cc2C1', 'Clc1cccc(Cc2ccccc2)c1', 'c1ccc2c(c1)Nc1ccccc1S2', 'CCNC(=O)Cc1ccc(Cl)c(Cl)c1', 'CCN(C=O)Cc1cnc(C)nc1N', 'OC(c1ccccc1)c1ccccc1', 'COc1ccc(S(=O)=O)cc1C(N)=O', 'C[C@H]1C[C@@]2(C)C=CC(=O)C=C2CC1', 'C(=Cc1ccccc1)c1ccccc1', 'CC[C@@H]1CC[C@](O)(C(=O)CO)C1C', 'CCC1C(=O)NCN1c1ccccc1', 'CC1=CC(=O)CC[C@]1(C)C[C@H](C)O', 'c1ccc(CCc2ccccc2)cc1', 'C=CCC1(CC)C(=O)NC(=O)NC1=O', 'Clc1ccc(Sc2ccccc2)cc1', 'Clc1ccc(Oc2ccccc2)cc1', 'CCC1=CC(=O)C=CC1C[C@H](C)O', 'Fc1ccc(Cc2ccccc2)cc1', 'CCCC(C)C(O)C(=O)COC(C)=O', 'c1ccc(Cc2ccccc2)cc1', 'CC[C@H](NC=O)c1ccccc1', 'CCC1(CC)C(=O)NC(=O)NC1=O', 'Cn1c(=O)c2ncnc2n(C)c1=O', 'c1ccc(CN2CCCCC2)cc1', 'C[C@H]1OC(C)(C)OC1C(=O)CO', 'CNC(=O)Cc1ccc(Cl)c(Cl)c1', 'CCC(c1cccc(O)c1)C(C)C', 'Clc1ccccc1Cc1ccsc1', 'CCCC(=O)c1ccc(F)cc1', 'CC1C=CC(=O)C=C1[C@H](C)F', 'O=c1ccnc2ccc(F)cc12', 'Clc1ccc2c(c1)=CNCCN=2', 'CC1CC[C@H]2OC(C)(C)OC12', 'O=c1nc2ccc(Cl)cc2[nH]1', 'CCC1COc2c(O)cccc21', 'CCCCOc1ccccc1O', 'CCN(C)CCc1ccccc1', 'Nc1nc(=O)c2ncnc2[nH]1', 'CC(C=CC(C)=O)C[C@H](C)O', 'CC(Cc1ccccc1)N(C)C', 'COc1ccc2cc[nH]c2c1', 'CCC1(C)C(=O)NC(=O)NC1=O', 'CC(=O)N(C)c1ccc(Cl)cc1', 'CCCN1CCN(CCO)CC1', 'COc1cccc(OC)c1OC', 'O=CC12CC3CC(CC(C3)C1)C2', 'O=C1CC2(CCCC2)CC(=O)N1', 'CCC(C=O)c1ccccc1', 'Oc1ccc2c(c1)CCCC2', 'CCCC(C)C(O)C(=O)CO', 'CCCC(O)c1ccccc1', 'COc1ccc(S(=O)=O)cc1', 'CCC1(C)CNC(=O)NC1=O', 'Fc1ccc2ncccc2c1', 'CCC(O)C(=O)COC(C)=O', 'CCCCc1ccc(F)cc1', 'CCC(C)c1cccc(O)c1', 'CC1=C(C)NC(C)=C(C=O)C1', 'CC(=O)Nc1ccc(O)cc1', 'Cc1c[nH]c2ccccc12', 'COc1ccc(C)cc1OC', 'CC1(C)C=CC(=O)C=C1CF', 'O=C1C=CC2CCCCC2=C1', 'CCCC1=CC(=O)C=CC1C', 'O=c1nc[nH]c2ncnc12', 'O=c1nc2ccccc2[nH]1', 'CC(=O)Nc1ccc(Cl)cc1', 'CCC1=CC(=O)C=CC1C', 'c1ccc2[nH]ccc2c1', 'CCNc1ccccc1S', 'FC(F)(F)c1ccccc1', 'c1ccc2ncccc2c1', 'COc1ccccc1OC', 'CN1CCN(CCO)CC1', 'CCCCN1CCNCC1', 'CCCCc1ccccc1', 'NC(=O)Cc1ccccc1', 'CCCC(=O)COC(C)=O', 'c1cnc2ncccc2c1', 'CCC1=CC(=O)CCC1C', 'CC[C@@H]1COC(C)(C)O1', 'c1ccc2ccccc2c1', 'COc1cccc(C=O)c1', 'CC1C=CC(=O)C=C1CF', 'CC(N)Cc1ccccc1', 'c1ccc2occcc2c1', 'NC(C=O)c1ccccc1', 'c1ccc2ncncc2c1', 'CNCCc1ccccc1', 'CC1C=CC(=O)C=C1CC', 'CN1CCC(C(N)=O)CC1', 'CN1CCC(CCO)CC1', 'CCC(C)C(CC)C(N)=O', 'CC(=O)Nc1ccccc1', 'CC1CC(O)CC(C)C1C', 'NC(=O)CN1CCCC1=O', 'CC(=O)C=CC(C)CCO', 'C1C2CC3CC1CC(C2)C3', 'C=CCC(C(N)=O)C(C)C', 'Cc1ncc(CS)cc1O', 'Nc1ncc2ncnc2n1', 'CCCC(=O)N(C)CC=O', 'c1ccc2[nH]cnc2c1', 'O=C(O)Cc1ccccc1', 'CCC1(C)C=CC(=O)C=C1', 'CCC12C=CC(CC1)CC2', 'FC(F)c1ccccc1', 'CC1C=CC(=O)C=C1C', 'O=S(=O)c1ccccc1', 'CC1CCC[C@@H](O)C1', 'c1ccc2nccc2c1', 'COc1ccccc1O', 'CCNc1ccccc1', 'O=CNc1ccccc1', 'c1ccc2c(c1)OCO2', 'CC(O)c1ccccc1', 'CC1=CC(=O)C=CC1C', 'CCN1CCN(C)CC1', 'CCC1C=CC(=O)C=C1', 'COc1ccc(C)cc1', 'cccc1ccccc1', 'Cn(c=O)c1cncn1', 'c1ccc2occc2c1', 'CN1CCN(C=O)CC1', 'NC(=O)c1ccccc1', 'CN1CCN(C=N)CC1', 'Cc1nccc(=O)c1O', 'CCC(C(N)=O)C(C)C', 'O=Cc1cccc(Cl)c1', 'c1ncc2ncnc2n1', 'CC(C)C(O)C(=O)CO', 'CCCCCCC(C)F', 'CCC(C)C[C@H](C)O', 'CCC1C=CCC(O)C1', '[nH]c1cccc(Cl)c1', 'OCCN1CCNCC1', 'c1ccn2ccnc2c1', 'CC1CC(=O)NC(=O)C1', 'cnc(=O)c1cncn1', 'CCCC(O)C(=O)CO', 'CCCN(CC)CCC', 'O=C(O)c1ccccc1', 'CCc1ccccc1C', 'c1cnc2nncc2c1', 'CN1CC(=O)NC(=O)C1', 'O=[N+]([O-])c1ccccc1', 'OCC1CCC(O)CO1', 'CCOC(O)C(Cl)(Cl)Cl', 'CC1(C)OC(=O)NC1=O', 'FCc1ccccc1', 'COc1ccccc1', 'ccc1ccccc1', 'CCc1ccccc1', 'O=Sc1ccccc1', 'CCN1CCCCC1', 'O=Cc1ccccc1', 'CN1CCN(C)CC1', 'CCN1CCNCC1', 'Clc1ccccc1Cl', 'Cc1cccc(Cl)c1', 'CNc1ccccc1', 'CCC(CC)C(N)=O', 'CC(=O)C(O)C(C)C', 'CCN1CCOCC1', 'CCCCCCCC', 'Cc1ncccc1O', 'CC1CCCCN1C', 'Cc1cccc(C)c1', 'OCc1ccccc1', 'CC(C)C[C@H](C)O', 'CCC(O)C(=O)CO', 'Oc1ccccc1O', 'Cc1nccc(N)n1', '[nH]c1ccccc1', 'O=C1CCCC(=O)N1', 'Cc1ccccc1Cl', 'CCC(=O)OCC=O', 'CCCC(C)CCO', 'NCc1ccccc1', 'N#Cc1ccccc1', 'Clc1cccc(Cl)c1', 'nc(=O)c1cncn1', 'CC(C)CC(=O)CO', 'OCC1CNCCO1', 'COC(O)C(Cl)(Cl)Cl', 'Cc1ncc(CN)n1', 'CC1C=CC(=O)C=C1', 'Cc1cccc(Br)c1', 'O=CN1CCNCC1', 'O=C(O)/C=C\\C(=O)O', 'CNC(=O)CC(C)C', 'O=CCCCC(=O)O', 'Oc1ccc(Cl)cc1', 'CCN1CCCC1C', 'CCNCC1CCC1', 'O=CN1CCCCC1', 'Cc1ccccc1', 'Clc1ccccc1', 'Oc1ccccc1', 'Sc1ccccc1', 'Fc1ccccc1', 'CN1CCNCC1', 'CN1CCCCC1', 'Nc1ccccc1', 'CCCC(C)CO', 'CC(=O)OCC=O', 'CC(=O)CC(C)C', 'CC1CCCCC1', 'CCN1CCCC1', 'Cc1ccccn1', 'CCC=CC(C)=O', 'CCN(CC)CC', 'CCCC(=O)CO', 'CCCCNCC', 'Cc1ncccn1', 'CCCCC(=O)O', 'CCNCC1CC1', 'CCCCCCC', 'cc1ccccc1', 'CC(C)(C)C(=O)O', 'CCOC(C)(C)O', 'CCCC(C)CC', 'CN(C)CCCN', 'CCNC(=O)CN', 'CC(C)CCC=O', 'CCCC(=O)NC', 'O=cc1cncn1', 'NCc1ncnn1', 'OC1CCCCC1', 'OCC1=NCCN1', 'CC(F)[C@H](C)O', 'O=C1CCC(=O)N1', 'CN1CC=CCC1', 'Cc1ccn[nH]1', 'Oc1cccnc1', 'O=CCCC(=O)O', 'CC(CO)CCO', 'CCCCN(C)C', 'Cn1cccnc1', 'C1CCCNCC1', 'OC1CCNCC1', 'cc1ccncn1', 'CCNC(=O)CCl', 'CCCNC(C)=O', 'CCN(C=O)CC', 'CC1CCN(C)C1', 'c1ccccc1', 'CCCC(C)C', 'c1ccncc1', 'C1CCNCC1', 'C1CNCCN1', 'CCCCCO', 'c1cncnc1', 'CC(C)CC=O', 'CCNC(C)=O', 'CC(C)CCC', 'CCCCC=O', 'CC(C)CCO', 'CCCC(C)=O', 'CCCNCC', 'CNC(=O)CN', 'CC[C@H](C)O', 'CC(C)C(=O)O', 'CCCCCC', 'OCC(Cl)(Cl)Cl', 'CCN(C)CC', 'CCCN(C)C', 'CCCC(N)=O', 'CCCNC=O', 'CCCC(C)O', 'O=C1CCCN1', 'CCCC(=O)O', 'CN(C)CCO', 'CCN(C)C=O', 'NC(=O)NC=O', 'CC(C)N(C)C', 'cc1cncn1', 'CCC(C)(C)O', 'c1cc[nH]c1', 'CN1CCCC1', 'c1ccnnc1', 'Cc1ccnn1', 'CC1CCCC1', 'CCCCNC', 'CCOC(C)=O', 'CCNC(C)C', 'CCCCCF', 'Cc1nccn1', 'CCOC(N)=O', 'CCNCCO', 'O=C1NCCO1', 'Cc1ccno1', 'O=CNCCCl', 'CCC(C)CC', 'c1c[nH]cn1', 'C=CCNCC', 'O=CCCC=O', 'c1cnccn1', 'Cc1cccs1', 'O=CCNC=O', 'CC(=O)NC=O', 'NC(=O)C(N)O', 'C1CCCCC1', 'O=C1NCCN1', 'CC(C)NC=O', 'CCCCC', 'CCCC=O', 'CCNCC', 'CCCCO', 'CCC(C)C', 'CCN(C)C', 'CCNC=O', 'c1cncn1', 'CC(C)CC', 'CCC(=O)O', 'OCC(Cl)Cl', 'c1ccsc1', 'CNC(C)=O', 'NCC(N)=O', 'CCC(C)=O', 'c1ncnn1', 'O=CNC=O', 'c1cnnc1', 'c1ccoc1', 'ccccn', 'CCC(N)=O', 'CCC(C)O', 'CNCCN', 'CCOC=O', 'CCCNC', 'CCCCN', 'ccccc', 'CNCC=O', 'C1CCNC1', 'COC(N)=O', 'CCS(=O)=O', 'Ccccn', 'OCC(F)F', 'CN(C)C=O', 'c1cscn1', 'CC(C)(O)O', 'OCCCS', 'c1ncon1', 'CC=CC=O', 'CC(C)(C)O', 'CCCCF', 'c1nncs1', 'O=CNC=S', 'COCCO', 'FCC(F)F', 'CC(C)CO', 'c1ccnc1', 'cccc', 'CCCC', 'CCNC', 'CC(C)=O', 'cccn', 'CC(N)=O', 'CC(=O)O', 'O=CCO', 'OCCCl', 'CNC=O', 'CC(C)O', 'CN(C)C', 'NCC=O', 'CC(C)N', 'CCCO', 'CCC=O', 'OCCO', 'Cnc=O', 'CC(C)C', 'NCCO', 'cncn', 'OCCF', 'cc[nH]', 'CCCN', 'FC(F)F', 'NC(N)=O', 'COC=O', 'CC(F)F', 'O=P(O)O', 'nc[nH]', 'NS(=O)=O', 'NCCN', 'ccc=O', 'NC(=O)O', 'CCCS', 'O=CCCl', 'C=COC', 'O=S(=O)O', 'C[N+](C)[O-]', 'CCN', 'CCO', 'CCC', 'NC=O', 'ccn', 'CNC', 'CC=O', 'nc=O', 'O=CO', 'ncn', 'O=S=O', 'CCF', 'ccc', 'FCF', 'O=[N+][O-]', 'ncs', 'COC', 'OPO', 'C[N+][O-]', 'CCS', 'ClCCl', 'C1CC1', 'CCCl', 'cc', 'CC', 'CN', 'cn', 'CO', 'O=S', 'CF', 'C=O', '[N+][O-]', 'CS', 'OP', 'CCl', 'NN', 'nn', 'H', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl', 'Ca', 'Br', 'I']

    print(f"smis_0:{smis_0}")
    print(f"smis_1:{smis_1}")



    vocab = {0: smis_0, 1: smis_1}
    print(f"  Vocab size - Class 0: {len(smis_0)}, Class 1: {len(smis_1)}")

    # 加载GNN
    print("\n4. Loading pre-trained GNN...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
    gnn.eval()
    print("  GNN loaded successfully")

    # 创建带掩码的训练数据集 (with caching)
    print("\n5. Creating dataset with subgraph masks...")

    # 生成缓存文件名（基于数据集名称和vocab内容的哈希）
    os.makedirs('cache', exist_ok=True)
    vocab_str = str(sorted(vocab[0])) + str(sorted(vocab[1]))
    vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:8]
    cache_train = f'cache/graph_train_data_{args.dataset.lower()}_{vocab_hash}_{args.threshold}.pkl' \
        if args.threshold != 0 \
            else f'cache/graph_train_data_{args.dataset.lower()}_{vocab_hash}.pkl'


    # 检查缓存是否存在
    if os.path.exists(cache_train):
        print(f"  Found cached dataset at {cache_train}")
        print("  Loading from cache...")
        with open(cache_train, 'rb') as f:
            cache_data = pickle.load(f)

        # 恢复数据
        train_dataset_with_masks = cache_data['dataset']
        # 重要：需要更新args.max_subgraph_nodes，因为GraphTrainData会修改它
        if 'max_subgraph_nodes' in cache_data:
            args.max_subgraph_nodes = cache_data['max_subgraph_nodes']
            print(f"  Restored max_subgraph_nodes: {args.max_subgraph_nodes}")


        print(f"  Loaded {len(train_dataset_with_masks)} graphs from cache")
    else:
        print(f"  No cache found, creating dataset from scratch...")
        print(f"  This may take a while...")

        # 创建数据集（耗时操作）
        train_dataset_with_masks = GraphTrainData(args, train_loader, gnn, vocab)

        # 保存到缓存
        print(f"  Saving dataset to cache: {cache_train}")
        cache_data = {
            'dataset': train_dataset_with_masks,
            'max_subgraph_nodes': args.max_subgraph_nodes,  # 保存修改后的值
            'vocab_hash': vocab_hash,
            'dataset_name': args.dataset.lower(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cache_train, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Cache saved successfully")
        print(f"  Total graphs: {len(train_dataset_with_masks)}")

    # 为验证集创建带掩码的数据集 (with caching)
    cache_val = f'cache/graph_val_data_{args.dataset.lower()}_{vocab_hash}.pkl'

    # 检查缓存是否存在
    if os.path.exists(cache_val):
        print(f"  Found cached validation dataset at {cache_val}")
        print("  Loading from cache...")
        with open(cache_val, 'rb') as f:
            cache_data_val = pickle.load(f)

        # 恢复数据
        val_dataset_with_masks = cache_data_val['dataset']
        print(f"  Loaded {len(val_dataset_with_masks)} graphs from cache")
    else:
        print(f"  No cache found, creating validation dataset from scratch...")
        print(f"  This may take a while...")

        # 创建验证集数据集（耗时操作）
        val_dataset_with_masks = GraphTrainData(args, val_loader, gnn, vocab)

        # 保存到缓存
        print(f"  Saving validation dataset to cache: {cache_val}")
        cache_data_val = {
            'dataset': val_dataset_with_masks,
            'max_subgraph_nodes': args.max_subgraph_nodes,
            'vocab_hash': vocab_hash,
            'dataset_name': args.dataset.lower(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cache_val, 'wb') as f:
            pickle.dump(cache_data_val, f)
        print(f"  Cache saved successfully")
        print(f"  Total graphs: {len(val_dataset_with_masks)}")

    # 创建DataLoader (使用标准PyTorch DataLoader而不是PyG的DataLoader)
    print("\n6. Creating masked data loader...")
    train_loader_masked = TorchDataLoader(
        train_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=True,
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
    model = MyExplainer(args, gnn).to(args.device)
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

    evaluation_metrics = evaluate(
        args=args,
        model=trained_model,
        gnn=gnn,
        data_loader=train_loader_masked,
    )
    print("\nEvaluation Results on Training Set:")
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

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
