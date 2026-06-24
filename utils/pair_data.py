import time
from typing import Optional

import networkx as nx
import numpy as np
import torch
from networkx.algorithms import isomorphism
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import subgraph as pyg_subgraph

from utils.chemistry import smarts_to_data
from utils.subgraph_utils import generate_node_mappings, to_nx
from utils.subgraph_method import PatternBank

from tqdm import tqdm
import igraph as ig

class MappedDataset(Dataset):
    def __init__(self, config, dataset, patterns: PatternBank,
                 pred_labels: Optional[list] = None,
                 pred_probs: Optional[list] = None,
                 gnn=None):
        """
        初始化GraphTrainData数据集

        Args:
            config: ExplainerConfig 实例
            dataset: 训练集PyG的DataLoader，用于加载图数据
            pred_labels: 预训练GNN的分类结果
            patterns: 频繁子图的字典（{0：patterns_0, 1: patterns_1}）
        """
        self.patterns_0 = patterns[0]
        self.patterns_1 = patterns[1]
        self.device = config.device
        self.graphs = []  # 存储所有单个图
        self.subgraphs = []  # 存储所有图的频繁子图
        self.labels = pred_labels  # 存储GNN预测的标签
        self.probs = pred_probs  # 存储GNN预测的概率
        self.sub_masks = []  # 预计算的频繁子图掩码
        self.dataset_name = config.dataset
        self.thresh = config.threshold
        self.config = config
        if self.labels is None and self.probs is None:
            self._predict_label(config, dataset, gnn)
        # 预处理：从dataloader中提取所有图并进行预测
        print("  Processing graphs and computing subgraph masks...")
        self._process_graphs(dataset)
        # 预计算所有频繁子图掩码
        print("  Precomputing subgraph masks...")
        self._precompute_masks()
        print(f"  Precomputation completed for {len(self.graphs)} graphs")
        print(f"  length of sub_masks: {len(self.sub_masks)}")
        print(f"  length of graphs: {len(self.graphs)}")
        print(f"  length of subgraphs: {len(self.subgraphs)}")

    def _process_graphs(self, dataset):
        for data in dataset:
            self.graphs.append(data)

    def _precompute_masks(self):
        """
        预计算所有图的频繁子图掩码
        这样在__getitem__时就不需要重复计算SMILES转换和子图匹配
        """
        for idx in tqdm(range(len(self.graphs)), desc="  Computing masks"):
            graph = self.graphs[idx]

            if self.labels[idx] == 0:
                patterns = self.patterns_0
            else:
                patterns = self.patterns_1

            graph_nx = to_networkx(graph, node_attrs=['x']).to_undirected()
            subgraph_nx = self._find_largest_subgraph(graph_nx, patterns)


            subgraph = from_networkx(subgraph_nx)
            node_mappings = generate_node_mappings(graph, subgraph)

            subgraph.node_mappings = torch.tensor(node_mappings)
            self.subgraphs.append(subgraph)

    def _graph_padding(self, num_nodes, graphs):
        padded_graphs = []
        for graph in graphs:
            current_num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
            new_graph = graph.clone()
            if current_num_nodes < num_nodes:
                padding_size = num_nodes - current_num_nodes
                feature_dim = self.config.x_dim
                zero_features = torch.zeros(padding_size, feature_dim, dtype=graph.x.dtype, device=graph.x.device)
                new_x = torch.cat([graph.x, zero_features], dim=0)
                new_graph.x = new_x
                new_graph.num_nodes = num_nodes
                mask = torch.cat([
                    torch.ones(current_num_nodes, dtype=torch.bool, device=graph.x.device),
                    torch.zeros(padding_size, dtype=torch.bool, device=graph.x.device)
                ])
            else:
                mask = torch.ones(current_num_nodes, dtype=torch.bool, device=graph.x.device)
            new_graph.real_mask = mask
            padded_graphs.append(new_graph)
        return padded_graphs

    def _find_largest_subgraph(self, graph_nx, patterns):
        """
        在 graph_nx 中查找与 patterns 匹配的最大子图（使用 igraph VF2）

        返回：从 graph_nx 中提取的子图（仍然是 networkx.Graph）
        """
        # 1. 先把当前大图 graph_nx 转成 igraph.Graph
        g_ig = nx_to_igraph(graph_nx)

        best_match_vertices = None  # 记录在大图中的匹配顶点（ig 的顶点 id）

        # 假设 patterns 已经按“从大到小”排序（你原来就是这么设计的）
        for pattern_nx in patterns:
            # 2. 每个 pattern 也转成 igraph.Graph
            p_ig = nx_to_igraph(pattern_nx)

            # 3. 用 VF2 搜索所有子图同构映射
            #    调用形式：target.get_subisomorphisms_vf2(pattern)
            #    返回的是一个 list，每个元素是一个长度为 vcount(pattern) 的顶点 id 列表
            mappings = g_ig.get_subisomorphisms_vf2(p_ig)

            if not mappings:
                # 这个 pattern 没有匹配，换下一个 pattern
                continue
            # else:
            #     print("  Found a match!")

            # 因为 patterns 假定已经按 size 从大到小排过，
            # 找到的第一个 pattern 就是“最大”的，直接拿这个匹配即可
            best_match_vertices = mappings[0]  # 比如 [3, 7, 10, 11] 这样的 igraph 顶点 id
            break

        # 4. 如果一个 pattern 都没匹配上，就退回原图
        if best_match_vertices is None:
            return graph_nx

        # 5. 把 igraph 的顶点 id 映射回原来 networkx 的 node id
        matched_nodes = [g_ig.vs[v]["orig_id"] for v in best_match_vertices]

        # 6. 用这些 node id 从 graph_nx 里抽子图，保持原有属性
        subgraph = graph_nx.subgraph(matched_nodes).copy()
        return subgraph

    def _predict_label(self, config, dataset, model):
        """使用预训练GNN模型预测图的标签和概率"""
        self.labels = []
        self.probs = []
        model = model.eval()
        with torch.no_grad():
            for data in dataset:
                data = data.to(config.device)
                out = model(data.x, data.edge_index, data.batch)
                self.probs.extend(out.softmax(dim=1))
                preds = out.argmax(dim=1).cpu()
                self.labels.extend(preds)




    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        获取第idx个图及其预计算的频繁子图掩码

        Returns:
            dict: {
                "graph": PyG Data对象,
                "sub_mask": 频繁子图的边掩码 (edge_mask)
            }
        """
        return {
            "graph": self.graphs[idx],
            # "sub_mask": self.sub_masks[idx],
            "subgraph": self.subgraphs[idx]
        }



def train_collate_fn(batch):
    """
    GraphTrainData专用的collate函数

    Args:
        batch: list of dicts, 每个dict包含 {"graph": Data, "sub_mask": edge_mask}

    Returns:
        dict: {
            'graphs': Batch对象（所有图的batch）,
            'sub_masks': list of edge_mask（每个图的频繁子图掩码）
        }
    """
    # 手动提取graphs和masks，避免PyG的默认collator处理
    graphs = []
    # sub_masks = []
    subgraphs = []

    for item in batch:
        graphs.append(item['graph'])
        # sub_masks.append(item['sub_mask'])
        subgraphs.append(item['subgraph'])

    for subgraph in subgraphs:
        if 'num_nodes' not in subgraph:
            subgraph.num_nodes = subgraph.x.size(0)

    # 使用Batch.from_data_list合并图
    batched_graphs = Batch.from_data_list(graphs)
    # batched_subgraphs = Batch.from_data_list(subgraphs)



    # 返回字典，sub_masks保持为列表
    return {
        'graphs': batched_graphs,
        'subgraphs': subgraphs,
        # 'subgraphs': batched_subgraphs
    }


def nx_to_igraph(g_nx: nx.Graph) -> ig.Graph:
    """
    把 networkx.Graph 转成 igraph.Graph，并在 vertex 属性里保存原始 node id
    """
    # 固定一个节点顺序，给每个 nx 节点分配一个连续的 0..n-1 下标
    g_nx = remove_self_loops(g_nx)
    nodes = list(g_nx.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}

    # 用这些下标来建 igraph 的边
    edges = [(node_index[u], node_index[v]) for u, v in g_nx.edges()]

    g_ig = ig.Graph(edges=edges, directed=g_nx.is_directed())
    # 把原始的 node id 存在属性里，后面再映射回来
    g_ig.vs["orig_id"] = nodes

    return g_ig

def remove_self_loops(g: nx.Graph) -> nx.Graph:
    """返回一个拷贝，并去掉所有自环边（u,u）"""
    g2 = g.copy()
    g2.remove_edges_from(nx.selfloop_edges(g2))
    return g2






