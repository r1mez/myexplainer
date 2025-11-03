import argparse
import hashlib
import pickle
import sys
import os

import networkx as nx

from evaluation import evaluate

sys.path.append("..")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utils import get_datasets, GraphPairData, custom_collate_fn, GraphTrainData, train_collate_fn
from models.myexplainer import MyExplainer
from gnns import Mutag_GCN
from utils.vis_utils import visualize_subgraph
from utils.graph_utils import data_to_mol, MUTAG_atom_map, extract_explanatory_subgraph, exclude_explanatory_subgraph
from rdkit.Chem.Draw import MolToImage
from rdkit import Chem

from utils import concat_graphs
from torch_geometric.data import Data, Batch

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Test MyExplainer on test dataset")
    parser.add_argument("--cuda", type=int, default=2, help="GPU device.")
    parser.add_argument("--dataset", type=str, default="mutag", help="Dataset name.")
    parser.add_argument("--model_path", type=str, default="param/myexplainer_subgraph_best.pt", help="Path to trained model.")
    parser.add_argument("--gnn_path", type=str, default="param/", help="GNN directory.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top similar graphs for pairing.")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for data extraction.")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save visualization results.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to visualize.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing.")
    parser.add_argument('--train_mode', type=bool, default=False, help='Current mode')

    # Model hyperparameters (must match training)
    parser.add_argument("--x_dim", type=int, default=14, help="Node feature dimension.")
    parser.add_argument("--h_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--z_dim", type=int, default=32, help="Latent dimension.")
    parser.add_argument("--u_dim", type=int, default=32, help="U dimension.")
    parser.add_argument("--edge_attr_dim", type=int, default=3, help="Edge attribute dimension.")
    parser.add_argument("--max_num_nodes", type=int, default=20, help="Maximum number of nodes.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=20, help="Maximum number of subgraph nodes.")     # 53, 20
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    parser.add_argument("--visualize", type=bool, default=False, help="Whether to visualize counterfactuals.")

    return parser.parse_args()


def dense_to_sparse_graph(x_dense, adj_dense, mask, threshold=0.5, device='cpu'):
    """
    Convert dense graph representation back to PyG Data format

    Args:
        x_dense: (batch_size, max_num_nodes, x_dim) - node features
        adj_dense: (batch_size, max_num_nodes, max_num_nodes) - adjacency matrix
        mask: (batch_size, max_num_nodes) - node mask
        threshold: threshold for edge existence
        device: torch device

    Returns:
        list of Data objects
    """
    from torch_geometric.data import Data

    batch_size = x_dense.size(0)
    graphs = []

    for b in range(batch_size):
        # Get valid nodes
        num_nodes = mask[b].sum().item()
        if num_nodes == 0:
            continue

        # Extract node features
        x = x_dense[b, :num_nodes, :]

        # Extract edges from adjacency matrix
        adj = adj_dense[b, :num_nodes, :num_nodes]
        adj_binary = (adj > threshold).float()

        # Get edge indices
        edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):  # Upper triangular to avoid duplicates
                if adj_binary[i, j] > 0:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add reverse edge for undirected graph

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()

        # Create edge attributes (assume single bond type for generated graphs)
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros((num_edges, 3), device=device)
        edge_attr[:, 0] = 1.0  # Single bonds

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    return graphs

def generate_counterfactuals(args, model, gnn, test_loader):
    """
    Generate counterfactual explanations for test dataset

    对测试集中的每个图，生成其反事实解释（翻转预测标签）

    Args:
        args: Arguments
        model: Trained MyExplainer model
        gnn: Pretrained GNN classifier
        test_loader: DataLoader for test dataset (带子图掩码的数据)

    Returns:
        results: List of dictionaries containing original graphs, generated graphs, and predictions
            -       'ori_graph'
                    'gen_graph'
                    'ori_pred'
                    'cf_pred'
                    'ori_gnn_pred'
                    'gen_gnn_pred'
    """


    model.eval()
    gnn.eval()

    results = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), desc="Generating Counterfactuals", total=len(test_loader)):
            # 1. 准备数据
            graphs_batch = batch['graphs'].to(args.device)
            batch_size = graphs_batch.num_graphs

            # 2. 使用GNN获取原始预测
            ori_pred_logits = gnn.get_pred(graphs_batch.x, graphs_batch.edge_index, graphs_batch.batch)[0]
            ori_pred = ori_pred_logits.argmax(dim=1)  # (batch_size,)

            # 3. 反事实标签：翻转预测
            cf_pred = 1 - ori_pred
            y_cf = cf_pred.float().unsqueeze(1)
            y = ori_pred.float().unsqueeze(1)

            # 4. 准备子图特征和邻接矩阵
            subgraph_x_list = []
            zero_x_template = torch.zeros(args.max_subgraph_nodes, args.x_dim, device=args.device)

            # 提取所有子图节点特征
            for b in range(batch_size):
                # 提取单个子图
                mask = (batch['subgraphs'].batch == b)
                num_nodes_tensor = mask.sum()

                subgraph_x = batch['subgraphs'].x[mask]

                # 使用clone而不是zeros创建新张量
                subgraph_x_padded = zero_x_template.clone()
                num_nodes_i = num_nodes_tensor.item()
                if num_nodes_i > 0:
                    subgraph_x_padded[:num_nodes_i] = subgraph_x

                subgraph_x_list.append(subgraph_x_padded)

            # 堆叠所有子图节点特征
            all_subgraph_x = torch.stack(subgraph_x_list, dim=0)  # (B, max_num_nodes, x_dim)

            all_subgraph_adj = to_dense_adj(
                batch["subgraphs"].edge_index,
                batch=batch["subgraphs"].batch,
                max_num_nodes=args.max_subgraph_nodes
            )
            all_subgraph_x, all_subgraph_adj = all_subgraph_x.to(args.device), all_subgraph_adj.to(args.device)

            # 5. 使用模型生成反事实图
            outputs = model(
                features=all_subgraph_x,
                adj=all_subgraph_adj,
                y_cf=y_cf
            )

            # 6. 使用concat_graphs拼接生成的子图和原图
            concated_graphs = concat_graphs(args, outputs, batch)

            # 7. 使用GNN对生成的图进行预测
            gen_pred_logits = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)[0]
            if batch_idx == 0:
                print(gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)[0][17])
            gen_pred = gen_pred_logits.argmax(dim=1)  # (batch_size,)

            # 8. 保存结果
            # 将batch拆分为单个图
            ori_graphs_list = graphs_batch.to_data_list()
            gen_graphs_list = concated_graphs.to_data_list()

            for i in range(batch_size):
                result = {
                    'ori_graph': ori_graphs_list[i].to('cpu'),
                    'gen_graph': gen_graphs_list[i].to('cpu'),
                    'ori_pred': ori_pred[i].item(),
                    'cf_pred': cf_pred[i].item(),
                    'ori_gnn_pred': ori_pred[i].item(),  # 原始预测
                    'gen_gnn_pred': gen_pred[i].item()   # 生成图的预测
                }
                results.append(result)

    # 新增：计算并输出翻转成功率（按类别分别计算）
    total_samples = len(results)
    successful_flips = sum(1 for r in results if r['gen_gnn_pred'] == r['cf_pred'])

    # 分别计算0->1和1->0的翻转成功率
    successful_0_to_1 = sum(1 for r in results if r['ori_pred'] == 0 and r['gen_gnn_pred'] == 1)
    successful_1_to_0 = sum(1 for r in results if r['ori_pred'] == 1 and r['gen_gnn_pred'] == 0)
    total_0_samples = sum(1 for r in results if r['ori_pred'] == 0)
    total_1_samples = sum(1 for r in results if r['ori_pred'] == 1)

    success_rate = (successful_flips / total_samples) * 100 if total_samples > 0 else 0
    success_rate_0_to_1 = (successful_0_to_1 / total_0_samples) * 100 if total_0_samples > 0 else 0
    success_rate_1_to_0 = (successful_1_to_0 / total_1_samples) * 100 if total_1_samples > 0 else 0

    print(f"Counterfactual Flip Success Rate: {success_rate:.2f}% ({successful_flips}/{total_samples})")
    print(f"0->1 Flip Success Rate: {success_rate_0_to_1:.2f}% ({successful_0_to_1}/{total_0_samples})")
    print(f"1->0 Flip Success Rate: {success_rate_1_to_0:.2f}% ({successful_1_to_0}/{total_1_samples})")




    return results


def visualize_counterfactuals(args, results, test_dataset, gnn, save_dir=None, max_samples=20, figsize=(12, 10)):
    """
        Visualize counterfactual explanations from generate_counterfactuals output.

        For each result, plot the original graph and the generated counterfactual graph side-by-side,
        with labels indicating predictions. Node labels are annotated with atom names based on one-hot encoded features.

        Args:
            results: List of dictionaries from generate_counterfactuals.
            test_dataset: Dataset corresponding to results, for subgraph info.
            save_dir: Directory to save visualization images (default: None, displays instead of saving).
            max_samples: Maximum number of samples to visualize (default: 20, for efficiency).
            figsize: Figure size for each subplot (default: (12, 5)).
            layout: Layout type (default: 'planar', currently unused).

        Returns:
            None: Saves or displays plots.
        """
    # Atom mapping for one-hot encoded node features
    atom_map = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
    device = args.device
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    num_samples = min(max_samples, len(results))

    for i in range(num_samples):
        result = results[i]
        ori_graph = result['ori_graph']
        gen_graph = result['gen_graph']

        exp_graph = extract_explanatory_subgraph(ori_graph, gen_graph)
        exp_excluded_graph = exclude_explanatory_subgraph(ori_graph, gen_graph)
        exp_graph_batch = Batch.from_data_list([exp_graph]).to(device)
        exp_excluded_graph_batch = Batch.from_data_list([exp_excluded_graph]).to(device)
        exp_pred = gnn.get_pred(exp_graph_batch.x, exp_graph_batch.edge_index, exp_graph_batch.batch)[0].argmax(dim=1).item()
        exp_excluded_pred = gnn.get_pred(exp_excluded_graph_batch.x, exp_excluded_graph_batch.edge_index, exp_excluded_graph_batch.batch)[0].argmax(dim=1).item()

        ori_pred = result['ori_pred']
        cf_pred = result['cf_pred']
        gen_gnn_pred = result['gen_gnn_pred']

        # Get the corresponding subgraph and node mappings for highlighting
        subgraph = test_dataset[i]['subgraph']
        node_mappings = subgraph['node_mappings']  # List of original graph node indices
        highlight_nodes = node_mappings  # Set for O(1) lookup

        # Convert PyTorch Geometric Data to NetworkX graphs
        G_ori = to_networkx(ori_graph, to_undirected=True)
        G_gen = to_networkx(gen_graph, to_undirected=True)
        G_exp = to_networkx(exp_graph, to_undirected=True)
        G_exp_excluded = to_networkx(exp_excluded_graph, to_undirected=True)

        # Extract atom names for node labels from one-hot features
        def get_atom_labels(graph_data, atom_map):
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                x = graph_data.x  # Node features: (num_nodes, num_features)
                atom_indices = torch.argmax(x, dim=1).cpu().numpy()  # Argmax to get atom index per node
                labels = {node: atom_map.get(idx.item(), f'Unknown({idx.item()})') for node, idx in enumerate(atom_indices)}
            else:
                # Fallback: use node index as string
                labels = {node: str(node) for node in graph_data.num_nodes}
            return labels

        ori_labels = get_atom_labels(ori_graph, atom_map)
        gen_labels = get_atom_labels(gen_graph, atom_map)
        exp_labels = get_atom_labels(exp_graph, atom_map)
        exp_excluded_labels = get_atom_labels(exp_excluded_graph, atom_map)

        # Compute positions using original graph layout (reuse for both)
        pos = nx.spring_layout(G_ori)


        missing_nodes = set(G_ori.nodes()) - set(pos.keys())
        if missing_nodes:
            print(f"Warning: Missing positions for nodes {missing_nodes}")
            # 可选：为缺失节点分配默认位置，例如使用 spring_layout 补全
            additional_pos = nx.spring_layout(G_ori.subgraph(missing_nodes))
            pos.update(additional_pos)


        # Create figure with two subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Plot original graph with highlighted nodes and atom labels
        node_colors_ori = ['red' if node in highlight_nodes else 'lightblue' for node in G_ori.nodes()]
        nx.draw(G_ori, pos, ax=ax1, labels=ori_labels, with_labels=True, node_color=node_colors_ori,
                node_size=100, font_size=10, font_weight='bold', edge_color='gray')
        ax1.set_title(f'Original Graph\nPred: {ori_pred} (Target CF: {cf_pred})')

        missing_nodes = set(G_gen.nodes()) - set(pos.keys())
        if missing_nodes:
            print(f"Warning: Missing positions for nodes {missing_nodes}")
            # 可选：为缺失节点分配默认位置，例如使用 spring_layout 补全
            additional_pos = nx.spring_layout(G_gen.subgraph(missing_nodes))
            pos.update(additional_pos)

        # Plot generated graph with highlighted nodes and atom labels, using same positions
        node_colors_gen = ['orange' if node in highlight_nodes else 'lightgreen' for node in G_gen.nodes()]
        nx.draw(G_gen, pos, ax=ax2, labels=gen_labels, with_labels=True, node_color=node_colors_gen,
                node_size=100, font_size=10, font_weight='bold', edge_color='gray')
        ax2.set_title(f'Counterfactual Graph\nGNN Pred: {gen_gnn_pred}')


        node_colors_exp = ['yellow' if node in highlight_nodes else 'gray' for node in G_exp.nodes()]
        nx.draw(G_exp, pos, ax=ax3, labels=exp_labels, with_labels=True, node_color=node_colors_exp,
                node_size=100, font_size=10, font_weight='bold', edge_color='gray')
        ax3.set_title(f'Explanatory Subgraph\nGNN Pred: {exp_pred}')

        node_colors_exp_excluded = ['magenta' if node in highlight_nodes else 'pink' for node in G_exp_excluded.nodes()]
        nx.draw(G_exp_excluded, pos, ax=ax4, labels=exp_excluded_labels, with_labels=True, node_color=node_colors_exp_excluded,
                node_size=100, font_size=10, font_weight='bold', edge_color='gray')
        ax4.set_title(f'Explanatory Subgraph Excluded\nGNN Pred: {exp_excluded_pred}')

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f'cf_sample_{i}.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)

    if save_dir:
        print(f"Visualizations saved to {save_dir}")
    else:
        print(f"Displayed visualizations for {num_samples} samples.")

def main():
    args = parse_args()

    # 设置设备为torch.device对象
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # 加载数据集
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    # vocab_len = 500
    # smis_0, _ = graph_bpe('data/mutag/smiles/smiles_0.txt', vocab_len, f'data/mutag/smiles_bpe_{vocab_len}_0.txt', 16, False)
    # smis_1, _ = graph_bpe('data/mutag/smiles/smiles_1.txt', vocab_len, f'data/mutag/smiles_bpe_{vocab_len}_1.txt', 16, False)
    # print("smis_0:", smis_0)
    # print("smis_1:", smis_1)
    # smis_0 = ['C', 'cc', 'cccc', 'O', 'N', 'CC', 'c1ccccc1', 'CCO', 'Cc1ccccc1', '[N+][O-]', 'O=[N+][O-]', 'CCN', 'cn', 'Cl', 'cccccccc', 'CO', 'Nc1ccccc1', 'CN', 'ccc1ccccc1', 'S', 'c1ccc2ccccc2c1', 'O=[N+]([O-])c1ccccc1', 'O=Cc1ccccc1', 'ccn', 'O=S', 'CCl', 'CC=O', 'CCNC', 'Br', 'CC1CO1', 'ccccC', 'H', 'Oc1ccccc1', 'CCCl', 'cccc1ccccc1', 'c1ccc2ncccc2c1', 'cccn', 'CCC', 'CC(N)=O', 'N=O', 'C1CO1', 'CC(=O)O', 'OCCO', 'CNC', 'O=S=O', 'Cc1ccccc1N', 'ccccc1ccccc1', 'c1ccncc1', 'CCc1ccccc1', 'CNc1ccccc1', 'ncn', 'O=C1c2ccccc2C(=O)c2ccccc21', 'cccccc', 'NC=O', 'O=[SH](=O)O', 'Nc1ccc2ccccc2c1', 'F', 'O=CO', 'O=C1c2ccccc2C(=O)c2c(O)cccc21', 'CCOCCO', 'CCCC', 'cccccccc[N+](=O)[O-]', 'Cc1ccccc1[N+](=O)[O-]', 'c1ccc2nc3ccccc3cc2c1', 'c1ccsc1', 'c1ccc2c(c1)ccc1ccccc12', 'ClCCl', 'Cc1ccco1', 'CC(O)CO', 'Cncn', 'NO', 'CCOP', 'c1ccc2[nH]ccc2c1', 'Nc1ccc([N+](=O)[O-])cc1', 'c1cncn1', 'ccccccc1ccccc1', 'CCCO', 'Cc1ccc([N+](=O)[O-])o1', 'P', 'c1ccc2cc3ccccc3cc2c1', 'CCNC=O', 'ccnc', 'Nc1cccc2ccccc12', 'Cc1ccc([N+](=O)[O-])cc1', 'O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'CC(C)O', 'Cnc(n)N', 'CCNCC', 'c1ccc(c2ccccc2)cc1', 'c1ccc2occcc2c1', 'nc1ccccc1', 'ccccn', 'c1cncnc1', 'cncn', 'CCNCCN', 'cc1ccccc1', 'Cc1ccc(N)cc1', 'ccc', 'N=Nc1ccccc1', 'I']
    # smis_1 = ['C', 'CC', 'cc', 'O', 'cccc', 'CCCC', 'N', 'c1ccccc1', 'CCC', 'CO', 'CCO', 'Cl', 'CN', 'cn', 'Cc1ccccc1', 'CCCCO', 'CCN', 'S', 'Oc1ccccc1', 'O=S', 'CCCCCCCC', 'O=CO', 'Clc1ccccc1', 'CCC=O', 'F', 'CC(=O)O', 'CCC(C)C', 'Br', 'O=S=O', 'CC=O', 'NC=O', 'Nc1ccccc1', 'CC(C)C', 'CCCC=O', 'cncn', 'CCCCC', 'ccn', 'H', 'CC(N)=O', 'O=Cc1ccccc1', 'CNC', 'CCCO', 'c1ccncc1', 'CCCC(=O)O', 'O=[SH](=O)O', 'P', 'CCC(C)O', 'OCCO', 'c[nH]', 'CCC(=O)O', 'COc1ccccc1', 'c1cncnc1', 'FCc1ccccc1', 'OP', 'ccccc', 'CCC(C)=O', 'Clc1cccc(Cl)c1', 'CC(O)CCO', 'O=C(O)c1ccccc1', 'Brc1ccccc1', 'OCC1OCC(O)CC1O', 'CCNCC', 'CC(C)O', 'CCNC', 'COP', 'CCCC(C)O', 'ccc1ccccc1', 'CC=CC', 'FC(F)c1ccccc1', 'NCCO', 'c1ccc2[nH]ccc2c1', 'Oc1ccccc1Br', 'c1ccc2occcc2c1', 'Oc1ccc(Cl)cc1', 'C[N+]', 'CCCC(C)C', 'CC(=O)Nc1ccccc1', 'c1ccc2ccccc2c1', 'N[SH](=O)=O', 'C=C(C)C=O', 'CC(O)CO', 'CCCCl', 'Cc1ccc(O)cc1', 'C1CCCCC1', 'CC(C)N', 'ccc', 'CCCCCC', 'CCCCC(=O)O', 'CCC(O)CO', 'c1ccc2ncccc2c1', 'CCC(N)=O', 'CC(=O)c1ccccc1', 'cccc1ccccc1', 'CC(C)=O', 'CCCCCC(C)CC', 'I', 'nc=O', '[N+]=O', 'O=[N+][O-]', 'CCc1ccccc1']
    # print(len(smis_0))
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
    #          'O=[N+]([O-])c1cccc2nc3ccccc3cc12', 'CN(C)c1ccc(N=Nc2ccccc2)cc1', 'Nc1ccc2c(S(=O)(=O)O)ccc(N)c2c1O',
    #          'O=[N+]([O-])c1cc2ccccc2c2ccccc12', 'c1ccc(-c2cocc2-c2ccccc2)cc1', 'O=[N+]([O-])c1c2ccccc2cc2ccccc12',
    #          'Cc1c(C)c2c(nc(N)n2C)c2nccnc12', 'O=[N+]([O-])c1cccc2c1ccc1ccccc12', 'O=C1c2ccccc2C(=O)c2ccccc21',
    #          'c1cc2ccc3cccc4ccc(c1)c2c34', 'O=c1c2ccccc2oc2cccc(O)c12', 'Cc1cc2c(nc(N)n2C)c2nccnc12',
    #          'cccc1ccc2cccc3cccc1c32', 'Cc1cc2nccnc2c2nc(N)n(C)c12', 'Cc1cc(-c2ccc(N)c(C)c2)ccc1N',
    #          'Cc1ccc(C)c2c1[nH]c1ccccc12', 'Cc1c2ccccc2c(C)c2ccccc12', 'CC(C)NC(=O)C=Cc1ccc([N+](=O)[O-])o1',
    #          'N=Nc1ccc(-c2ccc(N=N)cc2)cc1', 'Nc1ccc2cc(S(=O)(=O)O)cc(O)c2c1', 'OC1C=Cc2cc3ccccc3cc2C1O',
    #          'Oc1cccc2oc3ccccc3cc12', 'O=C(O)c1cnc2ccc(F)cc2c1=O', 'Cc1c2ccccc2cc2ccccc12',
    #          'c1ccc2c(c1)-c1ccccc1C1NC21', 'Nc1ccc2cc3ccccc3nc2c1', 'O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
    #          'Cc1nccc2[nH]c3ccccc3c12', 'CCNC(=O)C=Cc1ccc([N+](=O)[O-])o1', 'O=c1nnnc2c1[nH]c1ccccc12',
    #          'Cc1cccc2c1ccc1ccccc12', 'Cc1cccc2[nH]c3ccccc3c12', 'C[n+]1c2ccccc2cc2ccccc21', 'Nc1c2ccccc2nc2ccccc12',
    #          'O=c1c2ccccc2oc2ccccc12', 'O=[N+]([O-])c1ccc(c2ccccc2)cc1', 'Cc1cccc2ccc3ccccc3c12',
    #          'Oc1ccc2ccc3ccccc3c2c1', 'c1ccc2c(c1)ccc1ccccc12', 'c1ccc2nc3ccccc3cc2c1', 'c1ccc2cc3ccccc3cc2c1',
    #          'c1ccc2c(c1)[nH]c1ccccc12', 'Oc1ccccc1cc1ccccc1', 'c1ccc2c(c1)[nH]c1ccncc12', 'c1ccc(N=Nc2ccccc2)cc1',
    #          'Cn1cnc2c3cccnc3ccc21', 'c1ccc2c(c1)[nH]c1cnnnc12', 'c1ccc2c(c1)CCc1ccccc1-2', 'Nc1ccc2c(c1)Cc1ccccc1-2',
    #          'COC(=O)C=Cc1ccc([N+](=O)[O-])o1', 'COc1ccc2c(c1)OC1OC=CC21', 'O=c(c1ccccc1)c1ccccc1',
    #          'O=c1c(O)coc2cc(O)cc(O)c12', 'Nc1ccc(-c2ccc(N)cc2)cc1', 'Nc1ccc(Oc2ccccc2)cc1', 'COc1cc2ccncc2cc1OC',
    #          'c1ccc2c(c1)Cc1ccccc1-2', 'cccc1cccc2ccccc12', 'c1ccc(cc2ccccc2)cc1', 'Nc1ccc2cccc(N)c2c1O',
    #          'Nc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]', 'O=[N+]([O-])c1ccc2ccccc2c1', 'Cc1c([N+](=O)[O-])cccc1[N+](=O)[O-]',
    #          'c1ccc2c(c1)oc1ccccc12', 'O=c1ccoc2cc(O)cc(O)c12', 'CNc1ccc2nccnc2c1C', 'O=[N+]([O-])c1cccc2ccccc12',
    #          'c1ccc(Nc2ccccc2)cc1', 'COc1cccc2oc(=O)ccc12', 'Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]',
    #          'O=[N+]([O-])c1ccccc1SSCCl', 'Nc1ccc(-c2ccccc2)cc1', 'ccccccc1ccccc1',
    #          'O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'c1ccc(c2ccccc2)cc1', 'ccc1ccc2ccccc2c1', 'Nc1ccc2ccccc2c1O',
    #          'c1ccc(-c2ccccc2)cc1', 'ccc1cccc2ccccc12', 'c1ccc(c2cccnc2)cc1', 'O=c1ccnc2ccc(F)cc12',
    #          'Nc1c(O)ccc2ccccc12', 'CCOCCOCCOCCO', '[nH]c1ccc2ccccc2c1', 'Oc1cc(O)c2cccoc2c1', 'c1ccc(c2ccnnn2)cc1',
    #          'OCC1OC(O)C(O)C(O)C1O', 'Cc1cccc2nc(N)n(C)c12', 'C1=Cc2cccc3cccc1c23', 'Nc1ccc2ccccc2c1N',
    #          'O=CC=Cc1ccc([N+](=O)[O-])o1', 'O=Cc1ccc2[nH]cnc2c1', 'COc1ccc2c(c1)OCC2C', 'cc1ccc2ccccc2c1c',
    #          'CS(=O)(=O)Nc1ccc(N)cc1', 'CC(=O)c1ccc([N+](=O)[O-])cc1', 'ccccc1ccccc1cc', 'Cc1ccc(C)c2ccccc12',
    #          'Nc1ccc2cccc(O)c2c1', 'COc1ccc2ccncc2c1', 'nnc1c[nH]c2ccccc12', 'Nc1ccc2ccccc2c1', 'cccccccc[N+](=O)[O-]',
    #          'Nc1cccc2ccccc12', 'Cc1cccc2nccnc12', 'Cc1ccc2ccccc2c1', 'OCC1OCC(O)C(O)C1O', 'Cc1cccc2ccccc12',
    #          'CCc1cccc([N+](=O)[O-])c1', 'Fc1ccc2ncccc2c1', 'O=c1ccc2ccccc2o1', 'O=C1NC(=O)c2ccccc21',
    #          'c1ccc(-c2ccoc2)cc1', 'Oc1ccc2cccoc2c1', 'Cc1cnc2ccccc2n1', 'cccc(C)c1ccccc1', 'cc1ccc2ccccc2c1',
    #          'COc1nsc2ccccc12', 'Nc1ccc([N+](=O)[O-])cc1Cl', 'CCO[PH](=O)OC(C)CCl', 'cc1cccc2ccccc12',
    #          'O=Cc1cccc([N+](=O)[O-])c1', 'O=Nn1ccc2ccccc21', 'O=CCN=Cc1ccccc1', 'CCNC(=O)N(CCCl)N=O',
    #          'c1ccc(OCC2CO2)cc1', 'NC(=O)c1csc([N+](=O)[O-])c1', 'COc1cccc([N+](=O)[O-])c1', 'c1ccc2ccccc2c1',
    #          'c1ccc2ncccc2c1', 'ccccc1ccccc1', 'Cc1ccccc1[N+](=O)[O-]', 'c1ccc2[nH]ccc2c1', 'Nc1ccc([N+](=O)[O-])cc1',
    #          'c1ccc2occcc2c1', 'Cc1ccc([N+](=O)[O-])cc1', 'c1ccc2cnccc2c1', 'c1ccc2[nH]cnc2c1',
    #          'O=[N+]([O-])c1ccc(O)cc1', 'Nc1ncnc2ncnc12', 'Nc1ccccc1[N+](=O)[O-]', 'COc1cccc(C=O)c1', 'CCc1ccc(OC)cc1',
    #          'nc1cccc([N+](=O)[O-])c1', 'O=[N+]([O-])c1cccc(O)c1', 'O=C(O)c1ccccc1O', 'CCOc1ccc(N)cc1',
    #          'CN(C)c1ccc(N)cc1', 'CC(=O)Nc1ccccc1', 'O=[N+]([O-])c1ccccc1S', 'OCC1OCC(O)CC1O', 'CCN(CCCl)CCCN',
    #          'c1ccc2[nH]ncc2c1', 'CC1OCC(O)C(O)C1O', 'COc1ccccc1OC', 'OCCOc1ccccc1', 'Cn1cnc2ccccc21',
    #          'Clc1nsc2ccccc12', 'N=Cc1ccc([N+](=O)[O-])o1', 'c1nc2c[nH]cnc-2n1', 'O=[N+]([O-])c1ccccc1', 'cccc1ccccc1',
    #          'Cc1ccc([N+](=O)[O-])o1', 'c1ccc2sncc2c1', 'c1ccc(C2CN2)cc1', 'O=CNc1ccccc1', 'O=C(O)c1ccccc1',
    #          'CN(C)c1ccccc1', 'COc1ccccc1N', 'c1ccc(C2CO2)cc1', 'COc1ccccc1O', 'cc1ccccc1cC', 'Cc1ccc(N)cc1N',
    #          'c1ccc2nccc2c1', 'OCC1OCC(O)C1O', 'Cc1cc(N)ccc1N', 'ccccccccC', 'O=C(Cl)c1ccccc1', 'OC1COCC(O)C1O',
    #          'c1ncc2ncnc2n1', 'cccccccc=O', 'nncc1ccccc1', 'NC(=O)c1ccccc1', 'O=Cc1ccccc1O', 'CCO[PH](=O)OCC',
    #          'c1ccc2scnc2c1', 'CCNC(=O)CNC=O', 'CC(=O)c1ccccc1', 'ccccccccn', 'cccc1ccccn1', 'Cc1c(N)cccc1N',
    #          'Cc1cc(n)ccc1N', 'O=Cc1cccc(O)c1', 'Cc1ccc(N)c(C)c1', 'COc1ccc(C)cc1', 'CCNc1ccccc1', 'ccccc(C)ccc',
    #          'Cn1cncc1[N+](=O)[O-]', 'CCNC(=O)NCCCl', 'CCO[PH](=S)OCC', 'cccccccc', 'ccc1ccccc1', 'O=Cc1ccccc1',
    #          'Cc1ccccc1N', 'CCc1ccccc1', 'CNc1ccccc1', 'ncc1ccccc1', 'Cc1ccc(N)cc1', 'N=Nc1ccccc1',
    #          'O=[N+]([O-])c1cccs1', 'COc1ccccc1', 'cc1ccccc1c', 'cc1ccccc1n', 'Nc1ccc(N)cc1', 'N=Cc1ccccc1',
    #          'Cc1cccc(O)c1', 'ClCc1ccccc1', '[nH]c1ccccc1', 'CC1OCCCC1O', 'Nc1ccccc1O', 'OCc1ccccc1', 'Cc1cccc(N)c1',
    #          'Nc1ccc(O)cc1', 'oc1cccc(O)c1', 'CN(C)CCNC=O', 'O=cc1ccccc1', 'Nc1ccccc1Cl', 'nc1cccc(N)c1', 'cnc1ccccc1',
    #          'OCC(O)C(O)CO', 'Oc1ccccc1O', 'COC(C=O)=CC=O', 'O=[N+]([O-])c1cncn1', 'O=C1OC(O)C=C1Cl', 'Nc1ccccc1N',
    #          'Cc1ccc(F)cc1', '[O-][n+]c1ccccc1', 'Oc1ccc(Cl)cc1', 'OCC1OCCC1O', 'O=[N+]([O-])c1cncs1', 'Oc1ccc(O)cc1',
    #          'O=[N+]([O-])c1ccco1', 'Cc1ccccc1', 'Nc1ccccc1', 'Oc1ccccc1', 'cc1ccccc1', 'CCOPOCC', 'ClCCNCCCl',
    #          'OCCC(O)CO', 'Clc1ccccc1', 'Nc1ccncn1', 'nc1ccccc1', 'OCC(O)C1CO1', 'CN(C)CCCN', 'CCNCCCN', 'CN1CCNCC1',
    #          'cc1ncnc1n', 'nc1cccnc1', 'Cc1cccnc1', 'CO[PH](=O)OC', 'Cc1ccccn1', 'N=C1C=CCC=C1', 'Fc1ccccc1', 'NCCNCCN',
    #          'CO[PH](=S)OC', 'Nc1ccccn1', 'O=C1CCC(=O)N1', 'OC1C=COC1O', 'ClCC(Cl)=C(Cl)Cl', 'CCCOC(C)=O', 'CCCN(C)N=O',
    #          'c1ccccc1', 'c1ccncc1', 'cccccc', 'CCOCCO', 'Cc1ccco1', 'c1cncnc1', 'CCNCCN', 'CCNC(C)=O', 'C1COCCN1',
    #          'C1=CCC=CC1', 'C=CC(O)CO', 'CNCCCN', 'c1c[nH]cn1', 'CC(N)C(=O)O', 'Cn1ccnc1', 'CCOC(C)=O', 'O=CC=CC=O',
    #          'OCC(O)CO', 'OC1CC=CO1', 'OC1CCOC1', 'C1CNCCN1', 'O=C1NCCO1', 'O=CNCCCl', 'cccccn', 'ccccCBr',
    #          'CN(C=O)N=O', 'cncc[nH]', 'OCC(Br)CBr', 'O=NNCCO', 'cccccC', 'ccccC', 'O=[SH](=O)O', 'c1cncn1', 'c1ccsc1',
    #          'CC(O)CO', 'CCNCC', 'Cnc(n)N', 'CCNC=O', 'ccccn', 'O=C(O)CCl', 'c1cscn1', 'c1ccoc1', 'CCC(C)O',
    #          'C[SH](=O)=O', 'COPOC', 'CC(=O)NO', 'CNC(C)=O', 'CCNN=O', 'COC(C)=O', 'ccccc', 'Cc(n)cn', 'CCOP=O',
    #          'NC(=O)CBr', 'CCOCC', 'CCC(C)C', 'OCC1CO1', 'COCC=O', 'CCCNC', 'ClCC(Cl)Cl', 'CC(C)(C)O', 'C1=COCC1',
    #          'C=CC(=O)O', 'COC(N)=O', 'CCC(=O)O', 'OCCCBr', 'ClCCCCl', 'CNCCCl', 'ClCCCBr', 'CNC(=O)O', 'C1CSCN1',
    #          'CCC1CO1', 'CC(C)CO', 'C1CCCC1', 'OCC(Cl)Cl', 'ClCC1CO1', 'COcns', 'O=C(O)CBr', 'NC(=O)CCl', 'cccc',
    #          'CCNC', 'CC1CO1', 'cccn', 'CC(N)=O', 'CC(=O)O', 'OCCO', 'CCCO', 'CCCC', 'CCOP', 'Cncn', 'ccnc', 'CC(C)O',
    #          'nccn', 'NCCCl', 'O=CCO', 'O=CCCl', 'cncn', 'nc[nH]', 'CNN=O', 'OCCCl', 'ClCCCl', 'NCC=O', 'CCC=O', 'NCCO',
    #          'CC(C)N', 'C=CC=O', 'CNC=O', 'OCCBr', 'C=CCC', 'CCCCl', 'ClC(Cl)Cl', 'CC(C)C', 'CC(=O)Cl', 'CC(Cl)Cl',
    #          'SCCCl', 'O=CCBr', 'Cccn', 'BrCCBr', 'CC(C)Br', 'CN(C)N', 'O=CC=O', 'COC=O', 'CCO', 'O=[N+][O-]', 'CCN',
    #          'ccn', 'CC=O', 'CCC', 'C1CO1', 'CCCl', 'O=S=O', 'CNC', 'ncn', 'NC=O', 'O=CO', 'ClCCl', 'COP', 'ccc',
    #          'CCBr', 'OCO', 'SCCl', 'ncN', 'N=CN', 'nc=O', 'CC[N+]', 'cco', 'C[S+]C', 'C=CC', 'cns', 'NCN', 'ccs', 'cc',
    #          'CC', '[N+][O-]', 'cn', 'CO', 'CN', 'O=S', 'CCl', 'N=O', 'NO', 'C=O', 'NN', 'CS', '[n+][O-]', '[N+]=[N-]',
    #          'C#N', 'C[n+]', 'C[S+]', 'c=O', 'CBr', 'C[N+]', 'nn', 'C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I']
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
              'C1CO1', 'cc', 'CC', '[N+][O-]', 'cn', 'CO', 'CN', 'O=S', 'CCl', 'N=O', 'NO', 'H', 'B', 'C', 'N', 'O',
              'F',
              'Na', 'P', 'S', 'Cl', 'Ca', 'Br', 'I']
    smis_1 = ['CCCCCCCCCCCC', 'OCC1OCC(O)CC1O', 'c1ccc2[nH]ccc2c1', 'c1ccc2ncccc2c1', 'CC(=O)Nc1ccccc1',
              'c1ccc2ccccc2c1', 'O=C(O)c1ccccc1', 'FC(F)c1ccccc1', 'CCCCCCCCC', 'CCCCCCCC', 'O=Cc1ccccc1',
              'Clc1cccc(Cl)c1', 'ccc1ccccc1', 'FCc1ccccc1', 'COc1ccccc1', 'Cc1ccc(O)cc1', 'Oc1ccccc1Br', 'CCc1ccccc1',
              'Cc1ccccc1', 'Clc1ccccc1', 'Oc1ccccc1', 'Nc1ccccc1', 'Brc1ccccc1', 'CCCCC(=O)O', 'c1ccccc1', 'c1ccncc1',
              'CCCC(=O)O', 'c1cncnc1', 'CCOCCO', 'CC(O)CCO', 'CCCC(C)C', 'CCCC(C)O', 'CCC(O)CO', 'C=C(C)C(=O)O',
              'C1CCCCC1', 'CCCCO', 'CCC(C)C', 'CCCC=O', 'CCCCC', 'O=[SH](=O)O', 'CCC(C)O', 'CCC(=O)O', 'CCNCC',
              'CCC(C)=O', 'C=CC(=O)O', 'ccccc', 'COPOC', 'cccc', 'CCCC', 'CC(=O)O', 'cncn', 'CC(C)C', 'CC(C)O',
              'CC(N)=O', 'CCCO', 'OCCO', 'CCNC', 'CCC=O', 'NCCO', 'CC=CC', 'CCCCl', 'ccc=O', 'CCO', 'CCC', 'CCN',
              'O=CO','NC=O', 'O=S=O', 'CC=O', 'CNC', 'ccn', 'ccc', 'COP', 'c[nH]', 'CCF', 'O=[N+][O-]', 'CC', 'cc', 'CO', 'CN',
              'cn', 'O=S', 'OP', 'C=O', 'CS', '[N+][O-]', 'C[N+]', 'H', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl',
              'Ca', 'Br', 'I']

    # smis_0 = []
    # smis_1 = []
    # with open("smiles_0.txt","r") as file:
    #     smis_0 = [line.strip() for line in file.readlines()]
    # with open("smiles_1.txt", "r") as file:
    #     smis_1 = [line.strip() for line in file.readlines()]

    vocab = {0: smis_0, 1: smis_1}
    print(f"  Vocab size - Class 0: {len(smis_0)}, Class 1: {len(smis_1)}")

    # 加载GNN
    print("\n4. Loading pre-trained GNN...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
    gnn.eval()
    print("  GNN loaded successfully")

    # IMPORTANT: Initialize model BEFORE GraphTrainData to preserve max_subgraph_nodes
    # GraphTrainData will overwrite args.max_subgraph_nodes based on actual data,
    # but we need to use the value from training (53) for the checkpoint to load correctly
    print("\n5. Initializing MyExplainer model (BEFORE GraphTrainData)...")
    print(f"  Model config: x_dim={args.x_dim}, h_dim={args.h_dim}, z_dim={args.z_dim}")
    print(f"  max_num_nodes={args.max_num_nodes}, max_subgraph_nodes={args.max_subgraph_nodes}")
    print(f"  IMPORTANT: Model decoder dimensions = max_subgraph_nodes * x_dim = {args.max_subgraph_nodes} * {args.x_dim} = {args.max_subgraph_nodes * args.x_dim}")
    print(f"  IMPORTANT: Model decoder dimensions = max_subgraph_nodes^2 = {args.max_subgraph_nodes}^2 = {args.max_subgraph_nodes ** 2}")
    model = MyExplainer(args, gnn).to(args.device)

    # Save the max_subgraph_nodes value used for model initialization
    model_max_subgraph_nodes = args.max_subgraph_nodes

    # 创建带掩码的训练数据集 (WARNING: This will overwrite args.max_subgraph_nodes!)
    print("\n6. Creating training dataset with subgraph masks...")
    os.makedirs('cache', exist_ok=True)
    vocab_str = str(sorted(vocab[0])) + str(sorted(vocab[1]))
    vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:8]
    cache_test = f'cache/graph_test_data_{args.dataset.lower()}_{vocab_hash}.pkl'
    if os.path.exists(cache_test):
        print(f"  Found cached dataset at {cache_test}")
        print("  Loading from cache...")
        with open(cache_test, 'rb') as f:
            cache_data = pickle.load(f)

        # 恢复数据
        test_dataset_with_masks = cache_data['dataset']
        if 'max_subgraph_nodes' in cache_data:
            args.max_subgraph_nodes = cache_data['max_subgraph_nodes']
            print(f"  Restored max_subgraph_nodes: {args.max_subgraph_nodes}")

        print(f"  Loaded {len(test_dataset_with_masks)} graphs from cache")
    else:
        print(f"  No cache found, creating dataset from scratch...")
        print(f"  This may take a while...")

        # 创建数据集（耗时操作）
        test_dataset_with_masks = GraphTrainData(args, test_loader, gnn, vocab)

        # 保存到缓存
        print(f"  Saving dataset to cache: {cache_test}")
        cache_data = {
            'dataset': test_dataset_with_masks,
            'max_subgraph_nodes': args.max_subgraph_nodes,  # 保存修改后的值
            'vocab_hash': vocab_hash,
            'dataset_name': args.dataset.lower(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cache_test, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Cache saved successfully")
        print(f"  Total graphs: {len(test_dataset_with_masks)}")


    # 创建DataLoader (使用标准PyTorch DataLoader而不是PyG的DataLoader)
    print("\n7. Creating masked data loader...")
    test_loader_masked = TorchDataLoader(
        test_dataset_with_masks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total batches: {len(test_loader_masked)}")

    # Load trained model
    print(f"\nLoading trained model from {args.model_path}")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print("\nGenerating counterfactual explanations...")
    print("(自动为每个测试样本生成反事实解释，将预测标签翻转)")
    results = generate_counterfactuals(args, model, gnn, test_loader_masked)


    evaluation_metrics = evaluate(
        args=args,
        model=model,
        gnn=gnn,
        data_loader=test_loader_masked,
    )
    print("\nEvaluation Results on Testing Set:")
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
    if args.visualize:
        print("\nVisualizing counterfactual explanations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualize_counterfactuals(args, results, test_dataset_with_masks, gnn, save_dir= os.path.join(args.output_dir, timestamp))

if __name__ == "__main__":
    main()