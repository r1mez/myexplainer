import argparse
import hashlib
import pickle
import sys
import os

import networkx as nx

from evaluation import evaluate
from utils.batch_utils import core_data_from_batch
from utils.pair_data import GraphTrainDataBA2

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
from models.myexplainer import MyExplainer, MyExplainerBA2, MyCausalExplainer
from gnns import *
from utils.vis_utils import visualize_subgraph
from utils.graph_utils import data_to_mol, MUTAG_atom_map, extract_explanatory_subgraph, exclude_explanatory_subgraph, \
    process_outputs
from rdkit.Chem.Draw import MolToImage
from rdkit import Chem

from utils import concat_graphs
from torch_geometric.data import Data, Batch

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Test MyExplainer on test dataset")
    parser.add_argument("--cuda", type=int, default=2, help="GPU device.")
    parser.add_argument("--dataset", type=str, default="ba2motif", help="Dataset name.")
    parser.add_argument("--model_path", type=str, default="param/myexplainer_ba2motif_best.pt", help="Path to trained model.")
    parser.add_argument("--gnn_path", type=str, default="param/", help="GNN directory.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top similar graphs for pairing.")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for data extraction.")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save visualization results.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to visualize.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing.")
    parser.add_argument('--train_mode', type=bool, default=False, help='Current mode')

    # Model hyperparameters (must match training)
    parser.add_argument("--x_dim", type=int, default=10, help="Node feature dimension.")
    parser.add_argument("--h_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension.")
    parser.add_argument("--u_dim", type=int, default=32, help="U dimension.")
    parser.add_argument("--edge_attr_dim", type=int, default=0, help="Edge attribute dimension.")
    parser.add_argument("--max_num_nodes", type=int, default=25, help="Maximum number of nodes.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=25, help="Maximum number of subgraph nodes.")     # 53, 20
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # parser.add_argument("--visualize", type=bool, default=True, help="Whether to visualize counterfactuals.")
    parser.add_argument("--visualize", type=bool, default=True, help="Whether to visualize counterfactuals.")

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

            all_subgraph_x, all_subgraph_adj, all_subgraph_edge_attr = core_data_from_batch(args, batch)

            # 5. 使用模型生成反事实图
            outputs = model(
                x=all_subgraph_x,
                adj=all_subgraph_adj,
                edge_attr=all_subgraph_edge_attr,
                y_cf=y_cf
            )

            # 6. 使用concat_graphs拼接生成的子图和原图
            concated_graphs = concat_graphs(args, outputs, batch)

            # 7. 使用GNN对生成的图进行预测
            gen_pred_logits = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)[0]
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

def generate_counterfactuals_causality(args, model, gnn, test_loader):


    device = args.device
    model.eval()
    gnn.eval()

    results = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), desc="Generating Counterfactuals", total=len(test_loader)):
            graphs_batch = batch['graphs'].to(args.device)
            subgraphs_batch = batch['subgraphs'].to(args.device)

            batch_size = graphs_batch.num_graphs

            # 2. 使用GNN获取原始预测
            with torch.no_grad():
                ori_pred_logits = gnn.get_pred(graphs_batch.x, graphs_batch.edge_index, graphs_batch.batch)[0]
                ori_pred = ori_pred_logits.argmax(dim=1)  # (batch_size,)

            # 3. 反事实标签：翻转预测
            cf_pred = 1 - ori_pred
            y_cf = cf_pred.float().unsqueeze(1)
            y = ori_pred.float().unsqueeze(1)

            # 第一步：提取批量信息
            # 使用 to_dense_batch 将 [total_nodes, F] 转换为 [batch_size, max_num_nodes, F]
            all_graph_x, node_mask = to_dense_batch(graphs_batch.x, graphs_batch.batch, max_num_nodes=args.max_num_nodes)
            all_graph_x = all_graph_x.to(device)  # [batch_size, max_num_nodes, x_dim]

            # 将邻接矩阵转换为密集格式 [batch_size, max_num_nodes, max_num_nodes]
            all_graph_adj = to_dense_adj(graphs_batch.edge_index, graphs_batch.batch, max_num_nodes=args.max_num_nodes).to(device)

            # 提取子图数据
            all_subgraph_x, all_subgraph_adj, _ = core_data_from_batch(args, batch)

            if args.edge_attr_dim != 0:
                # 如果有边特征，使用 to_dense_adj 的 edge_attr 参数
                all_graph_edge_attr = to_dense_adj(
                    graphs_batch.edge_index,
                    graphs_batch.batch,
                    edge_attr=graphs_batch.edge_attr,
                    max_num_nodes=args.max_num_nodes
                ).to(device)

                outputs = model(
                    x=all_graph_x,
                    adj=all_graph_adj,
                    y_cf=y_cf,
                    edge_attr=all_graph_edge_attr,
                    x_sub=all_subgraph_x,
                    adj_sub=all_subgraph_adj
                )
            else:
                outputs = model(
                    x=all_graph_x,
                    adj=all_graph_adj,
                    y_cf=y_cf,
                    x_sub=all_subgraph_x,
                    adj_sub=all_subgraph_adj
                )

            # 第二步：使用批量输出结果中的每一个重构图
            output_graphs_batch = process_outputs(args, outputs)

            # 7. 使用GNN对生成的图进行预测
            gen_pred_logits = gnn.get_pred(output_graphs_batch.x, output_graphs_batch.edge_index, output_graphs_batch.batch)[0]
            gen_pred = gen_pred_logits.argmax(dim=1)  # (batch_size,)

            # 8. 保存结果
            # 将batch拆分为单个图
            ori_graphs_list = graphs_batch.to_data_list()
            gen_graphs_list = output_graphs_batch.to_data_list()

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
                labels = {}
                for node, idx in enumerate(atom_indices):
                    if x[node].sum() == 0:  # 如果节点特征全为0，则标签为'X'
                        labels[node] = 'X'
                    else:
                        labels[node] = atom_map.get(idx.item(), f'Unknown({idx.item()})')
            else:
                # Fallback: use node index as string
                labels = {node: str(node) for node in range(graph_data.num_nodes)}  # 修正：使用 range(graph_data.num_nodes)
            return labels

        ori_labels = get_atom_labels(ori_graph, atom_map)
        gen_labels = get_atom_labels(gen_graph, atom_map)
        exp_labels = get_atom_labels(exp_graph, atom_map)
        exp_excluded_labels = get_atom_labels(exp_excluded_graph, atom_map)

        # Compute positions using original graph layout (reuse for both)
        pos = nx.spring_layout(G_ori, seed=42)


        missing_nodes = set(G_ori.nodes()) - set(pos.keys())
        if missing_nodes:
            print(f"Warning: Missing positions for nodes {missing_nodes}")
            # 可选：为缺失节点分配默认位置，例如使用 spring_layout 补全
            additional_pos = nx.spring_layout(G_ori.subgraph(missing_nodes), seed=42)
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
            additional_pos = nx.spring_layout(G_gen.subgraph(missing_nodes), seed=42)
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

        # plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f'cf_sample_{i}.png'), dpi=300, bbox_inches='tight')
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
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载被解释GNN
    print("\n2. Loading pre-trained GNN classifier...")
    gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)
    gnn.eval()
    print("  GNN loaded successfully")

    # IMPORTANT: Initialize model BEFORE GraphTrainData to preserve max_subgraph_nodes
    # GraphTrainData will overwrite args.max_subgraph_nodes based on actual data,
    # but we need to use the value from training (53) for the checkpoint to load correctly
    print("\n3. Initializing MyExplainer model (BEFORE GraphTrainData)...")
    print(f"  Model config: x_dim={args.x_dim}, h_dim={args.h_dim}, z_dim={args.z_dim}")
    print(f"  max_num_nodes={args.max_num_nodes}, max_subgraph_nodes={args.max_subgraph_nodes}")

    model = MyExplainerBA2(args, gnn).to(args.device)
    # model = MyCausalExplainer(args, gnn).to(args.device)

    # Save the max_subgraph_nodes value used for model initialization
    model_max_subgraph_nodes = args.max_subgraph_nodes

    # 创建带掩码的训练数据集 (WARNING: This will overwrite args.max_subgraph_nodes!)


    print("\n4. Loading or mining frequent subgraph patterns...")
    patterns_0_path = f'fsm_results/{args.dataset}_0_patterns.pkl'
    patterns_1_path = f'fsm_results/{args.dataset}_1_patterns.pkl'
    with open(patterns_0_path, 'rb') as f:
        patterns_0 = pickle.load(f)
    with open(patterns_1_path, 'rb') as f:
        patterns_1 = pickle.load(f)

    def reverse_groups_new(lst, group_size=3):
        """
        返回新列表，按组倒序。
        """
        n = len(lst) // group_size
        result = []
        for i in range(n - 1, -1, -1):  # 从后往前遍历组索引
            result.extend(lst[i * group_size:(i + 1) * group_size])
        return result

    patterns_0 = reverse_groups_new(patterns_0, group_size=3)
    patterns_1 = reverse_groups_new(patterns_1, group_size=3)
    patterns = {0: patterns_0, 1: patterns_1}

    print("\n5. Creating test dataset with subgraph masks...")

    test_dataset_with_masks = GraphTrainDataBA2(args, test_dataset, patterns, gnn=gnn)


    print("\n6. Creating masked data loader...")
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
    # results = generate_counterfactuals_causality(args, model, gnn, test_loader_masked)


    # evaluation_metrics = evaluate(
    #     args=args,
    #     model=model,
    #     gnn=gnn,
    #     data_loader=test_loader_masked,
    # )
    # print("\nEvaluation Results on Testing Set:")
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
    #
    # print("  Robust Fidelity+ Prob ↑: {:.4f}".format(
    #     evaluation_metrics["ro_fid_prob_plus"]
    #     )
    # )
    # print("  Robust Fidelity- Prob ↓: {:.4f}".format(
    #     evaluation_metrics["ro_fid_prob_minus"]
    #     )
    # )
    # print("  Robust Fidelity Delta Prob ↑: {:.4f}".format(
    #     evaluation_metrics["ro_fid_prob_delta"]
    #     )
    # )
    # print("  Robust Fidelity+ Acc ↑: {:.4f}".format(
    #     evaluation_metrics["ro_fid_acc_plus"]
    #     )
    # )
    # print("  Robust Fidelity- Acc ↓: {:.4f}".format(
    #     evaluation_metrics["ro_fid_acc_minus"]
    #     )
    # )
    # print("  Robust Fidelity Delta Acc ↑: {:.4f}".format(
    #     evaluation_metrics["ro_fid_acc_delta"]
    #     )
    # )
    #
    #
    # print(
    #     "  Sparsity ↑: {:.4f}".format(
    #         evaluation_metrics["sparsity"]
    #     )
    # )
    if args.visualize:
        print("\nVisualizing counterfactual explanations...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualize_counterfactuals(args, results, test_dataset_with_masks, gnn, save_dir= os.path.join(args.output_dir, timestamp))

if __name__ == "__main__":
    main()