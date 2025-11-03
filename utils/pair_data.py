import time

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx.algorithms import isomorphism
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
from torch_geometric.utils import subgraph as pyg_subgraph

from data.mutag.smiles import data_to_smiles
from utils import find_largest_subgraph, generate_subgraph_mask
from utils.graph_utils import smarts_to_data
from utils.subgraph_utils import generate_node_mappings, to_nx

from tqdm import tqdm

class GraphPairData(Dataset):
    def __init__(self, args, dataloader, model, k=1):
        self.dataloader = dataloader  # pyg的DataLoader
        self.model = model.eval()  # 训练好的GNN模型，确保评估模式
        self.k = k  # 每个图配对的最近邻数量
        self.threshold = args.threshold  # 过滤预测概率的阈值
        self.graphs = []  # 存储所有图数据对象
        self.embeddings = []  # 存储图的embedding (通过get_graph_rep获取)
        self.pred_probs = []  # 存储图的预测概率分布
        self.pred_labels = []
        self.ystars = []  # 每个图的目标优质标签（预测概率第二高的标签）
        self.label_group = {}  # 按标签分组的图索引：{label: [graph_idx1, graph_idx2, ...]}
        self.device = args.device
        # 预处理：收集图数据、计算embedding和目标标签
        self._process_graphs()
        # 按标签分组
        self._group_by_label()
        # 创建配对索引
        self.create_pairs()

    def _process_graphs(self):
        """收集所有图数据并计算embedding和目标标签"""
        with torch.no_grad():  # 关闭梯度计算
            for data in self.dataloader:
                x = data.x
                edge_index = data.edge_index
                batch_idx = data.batch
                edge_attr = data.edge_attr
                x, edge_index, batch_idx, edge_attr = x.to(self.device), edge_index.to(self.device), batch_idx.to(self.device), edge_attr.to(self.device)
                # 获取图嵌入和预测概率
                graph_reps = self.model.get_graph_rep(x, edge_index, batch_idx)  # 假设模型有此方法返回embedding
                pred_probs, _ = self.model.get_pred(x, edge_index, batch_idx)

                # 逐个处理batch中的图
                num_graphs = data.num_graphs
                for i in range(num_graphs):
                    # 提取单个图的数据
                    mask = (batch_idx == i)
                    node_x = x[mask]  # 当前图的节点特征

                    # 提取单个图的边（仅保留源节点属于当前图的边）
                    edge_mask = mask[edge_index[0]]  # 边的源节点是否属于当前图
                    graph_edge_index = edge_index[:, edge_mask]
                    graph_edge_attr = edge_attr[edge_mask]

                    # 将边索引重新映射为局部索引（重要！单图应使用局部节点编号）
                    # 原边索引是基于整个batch的全局编号，转换为0开始的局部编号
                    unique_nodes = torch.unique(graph_edge_index)
                    node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}
                    # 重新映射边索引
                    graph_edge_index = torch.tensor(
                        [
                            [node_mapping[old.item()] for old in graph_edge_index[0]],
                            [node_mapping[old.item()] for old in graph_edge_index[1]]
                        ],
                        device=self.device,
                        dtype=torch.long  # 确保边索引为长整型
                    )

                    # 构建torch_geometric.data.Data对象（核心修改）
                    graph_data = Data(
                        x=node_x,  # 节点特征
                        edge_index=graph_edge_index,  # 边索引（已转为局部索引）
                        edge_attr=graph_edge_attr,  # 边特征
                        batch=batch_idx[mask]  # 保留batch信息（单图可省略）
                    )
                    # graph_data.smiles = data_to_smiles(graph_data)  # 可选：添加SMILES字符串等信息

                    # 可选：添加原始数据中的其他属性（如真实标签、原始索引等）
                    if hasattr(data, 'y'):
                        graph_data.y = data.y[i].clone().unsqueeze(0) # 真实标签
                    if hasattr(data, 'idx'):
                        graph_data.idx = data.idx[i].clone().unsqueeze(0)  # 原始数据索引
                    self.graphs.append(graph_data)

                    # 存储对应图的embedding和概率
                    self.embeddings.append(graph_reps[i])
                    self.pred_probs.append(pred_probs[i])

                    pred_label = torch.argmax(pred_probs[i]).item()  # argmax取概率最高的标签
                    self.pred_labels.append(pred_label)

                    # 计算目标优质标签（预测概率第二高的标签）
                    sorted_probs, sorted_labels = torch.sort(pred_probs[i], descending=True)
                    # 处理标签数量不足2的情况（实际Mutag数据集是2分类，这里做兼容）
                    ystar = sorted_labels[1].item() if sorted_probs.numel() >= 2 else sorted_labels[0].item()
                    self.ystars.append(ystar)

                # 将embedding列表转换为张量（便于批量计算）
            self.embeddings = torch.stack(self.embeddings, dim=0)

    def _group_by_label(self):
        """按标签对图索引进行分组"""
        for graph_idx, pred_label in enumerate(self.pred_labels):
            if pred_label not in self.label_group:
                self.label_group[pred_label] = []
            self.label_group[pred_label].append(graph_idx)

    def create_pairs(self):
        """创建源图与目标优质图的配对索引，优化版本（批量操作提升性能）"""
        self.pair_idxs = []  # 存储每个源图的k个配对索引

        # 预先将预测概率和标签转换为张量以加速批量操作
        pred_probs_tensor = torch.stack(self.pred_probs)  # 形状: [num_graphs, num_classes]
        pred_labels_tensor = torch.tensor(self.pred_labels, dtype=torch.long, device=self.device)  # 形状: [num_graphs]

        for src_idx in range(len(self.graphs)):
            # 源图的目标标签
            src_ystar = self.ystars[src_idx]

            # 获取目标标签对应的所有图索引
            if src_ystar not in self.label_group:
                self.pair_idxs.append([])
                continue
            tgt_indices = self.label_group[src_ystar]

            # 如果没有候选图，直接记录空列表
            if not tgt_indices:
                self.pair_idxs.append([])
                continue

            # 批量处理：将候选图索引转换为张量（在设备上操作）
            tgt_indices_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=self.device)

            # 批量获取候选图的预测概率（使用高级索引，一次操作完成）
            # 选取每个候选图对应其预测标签的概率
            tgt_probs = pred_probs_tensor[tgt_indices_tensor, pred_labels_tensor[tgt_indices_tensor]]

            # 批量过滤：找到概率 >= 阈值的候选图索引（设备上完成比较）
            mask = tgt_probs >= self.threshold
            # 将符合条件的索引映射回原始列表
            filtered_tgt_indices = [tgt_indices[i] for i in torch.where(mask)[0].cpu().numpy()]

            # 如果过滤后没有候选图，记录空列表
            if not filtered_tgt_indices:
                self.pair_idxs.append([])
                continue

            # 计算源图与过滤后的所有目标图的余弦距离
            src_emb = self.embeddings[src_idx].unsqueeze(0)  # 形状：[1, emb_dim]
            tgt_embs = self.embeddings[filtered_tgt_indices]  # 形状：[num_filtered_tgt, emb_dim]
            cos_sim = F.cosine_similarity(src_emb, tgt_embs, dim=1)  # 余弦相似度
            distances = 1 - cos_sim  # 转换为距离（值越小越相似）

            # 筛选最近的k个目标图
            num_tgt = len(filtered_tgt_indices)
            if num_tgt <= self.k:
                # 若目标图数量不足k，取全部
                topk_indices = torch.arange(num_tgt)
            else:
                # 取距离最小的k个
                _, topk_indices = torch.topk(distances, k=self.k, largest=False)

            # 映射回原始图索引并保存
            paired_indices = [filtered_tgt_indices[i] for i in topk_indices.cpu().numpy()]
            self.pair_idxs.append(paired_indices)

    def __len__(self):
        """总配对数 = 所有源图的配对数之和"""
        return sum(len(pairs) for pairs in self.pair_idxs)

    def __getitem__(self, idx):
        """获取第idx个配对数据"""
        # 找到对应的源图索引和配对索引
        src_idx = 0
        while idx >= len(self.pair_idxs[src_idx]):
            idx -= len(self.pair_idxs[src_idx])
            src_idx += 1
        pair_idx = self.pair_idxs[src_idx][idx]

        src_emb = self.embeddings[src_idx]
        tgt_emb = self.embeddings[pair_idx]
        cos_sim = F.cosine_similarity(src_emb.unsqueeze(0), tgt_emb.unsqueeze(0)).item()

        return {
            "ori_graph": self.graphs[src_idx],  # 源图数据（pyg的Data对象）
            "ori_pred": self.pred_labels[src_idx],  # 源图的预测标签
            "ori_emb": src_emb,  # 源图的embedding
            "tgt_graph": self.graphs[pair_idx],  # 配对的目标图数据
            "tgt_pred": self.pred_labels[pair_idx],  # 配对图的预测标签
            "tgt_emb": tgt_emb,  # 配对图的embedding
            "distance": 1 - cos_sim,  # 源图与目标图的余弦距离
        }


# import multiprocessing as mp
# from functools import partial
# from torch_geometric.data import Data
# import torch
# import networkx as nx
# from networkx.algorithms import isomorphism
# import os


# class GraphTrainData(Dataset):
#     """
#     用于训练的图数据集类
#
#     从dataloader中提取图数据，并为每个图找到对应类别的最大频繁子图掩码
#     """
#
#     def __init__(self, args, dataloader, gnn, vocab):
#         """
#         初始化GraphTrainData数据集
#
#         Args:
#             dataloader: PyG的DataLoader，用于加载图数据
#             gnn: 训练好的GNN分类器，用于预测图的类别
#             vocab: 频繁子图的SMILES字典
#                    格式: {0: [smiles1, smiles2, ...], 1: [smiles3, smiles4, ...]}
#                    key为类别标签，value为该类别的频繁子图SMILES列表
#             device: 设备 (cpu or cuda)
#         """
#         self.gnn = gnn.eval()  # 设置为评估模式
#         self.vocab = vocab
#         self.device = args.device
#         self.graphs = []  # 存储所有单个图
#         self.subgraphs = []  # 存储所有图的频繁子图
#         self.pred_labels = []  # 存储GNN预测的标签
#         self.sub_masks = []  # 预计算的频繁子图掩码
#
#         # 预处理：从dataloader中提取所有图并进行预测
#         print("  Processing graphs and computing subgraph masks...")
#         self._process_graphs(dataloader)
#         # Move all graphs to CPU to avoid sharing CUDA tensors in multiprocessing
#         print("  Moving graphs to CPU...")
#         for i in range(len(self.graphs)):
#             self.graphs[i] = self._move_to_cpu(self.graphs[i])
#         # 预计算所有频繁子图掩码
#         print("  Precomputing subgraph masks...")
#         self._precompute_masks()
#         print(f"  Precomputation completed for {len(self.graphs)} graphs")
#         print(f"  length of sub_masks: {len(self.sub_masks)}")
#         print(f"  length of graphs: {len(self.graphs)}")
#         print(f"  length of subgraphs: {len(self.subgraphs)}")
#         # 向args添加子图集合中最大原子数
#         args.max_subgraph_nodes = max([g.num_nodes for g in self.subgraphs]) + 3 if self.subgraphs else 0
#         self.max_subgraph_nodes = args.max_subgraph_nodes
#         print(f"  Max nodes in subgraphs: {args.max_subgraph_nodes}")
#         print(f"Padding subgraph with {self.max_subgraph_nodes}")
#         self.subgraphs = self._graph_padding(self.max_subgraph_nodes, self.subgraphs)
#
#     def _move_to_cpu(self, data):
#         """Move PyG Data object to CPU"""
#         data.x = data.x.cpu()
#         data.edge_index = data.edge_index.cpu()
#         if hasattr(data, 'edge_attr') and data.edge_attr is not None:
#             data.edge_attr = data.edge_attr.cpu()
#         if hasattr(data, 'y') and data.y is not None:
#             data.y = data.y.cpu()
#         # Remove batch if present, as it's single graph
#         if hasattr(data, 'batch'):
#             delattr(data, 'batch')
#         return data
#
#     def _process_graphs(self, dataloader):
#         """从dataloader中提取图并使用GNN进行预测"""
#         with torch.no_grad():
#             for data in dataloader:
#                 x = data.x.to(self.device)
#                 edge_index = data.edge_index.to(self.device)
#                 batch_idx = data.batch.to(self.device)
#                 edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None
#
#                 # 使用GNN进行预测
#                 pred_probs, _ = self.gnn.get_pred(x, edge_index, batch_idx)
#
#                 # 逐个处理batch中的图
#                 num_graphs = data.num_graphs
#                 for i in range(num_graphs):
#                     # 提取单个图的数据
#                     mask = (batch_idx == i)
#                     node_x = x[mask]
#
#                     # 提取单个图的边
#                     edge_mask = mask[edge_index[0]]
#                     graph_edge_index = edge_index[:, edge_mask]
#                     graph_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
#
#                     # 重新映射边索引为局部索引
#                     unique_nodes = torch.unique(graph_edge_index)
#                     node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}
#                     graph_edge_index = torch.tensor(
#                         [
#                             [node_mapping[old.item()] for old in graph_edge_index[0]],
#                             [node_mapping[old.item()] for old in graph_edge_index[1]]
#                         ],
#                         device=self.device,
#                         dtype=torch.long
#                     )
#
#                     # 构建单个图的Data对象
#                     graph_data = Data(
#                         x=node_x,
#                         edge_index=graph_edge_index,
#                         edge_attr=graph_edge_attr
#                     )
#
#                     # 保留原始标签（如果有）
#                     if hasattr(data, 'y'):
#                         graph_data.y = data.y[i].clone() if data.y.dim() > 0 else data.y.clone()
#
#                     self.graphs.append(graph_data)
#
#                     # 保存预测标签
#                     pred_label = torch.argmax(pred_probs[i]).item()
#                     self.pred_labels.append(pred_label)
#
#     @staticmethod
#     def _compute_single_mask(idx, graphs, pred_labels, vocab):
#         """
#         计算单个图的频繁子图
#         用于多进程并行计算
#         """
#         graph = graphs[idx]
#         pred_label = pred_labels[idx]
#         category_smiles = vocab.get(pred_label, [])
#         invalid_sub = set()
#
#         # 只转换一次原图
#         graph_nx = to_nx(graph)
#
#         while True:
#             subgraph, largest_subgraph_smiles = find_largest_subgraph(category_smiles, graph, invalid_sub)
#
#             if largest_subgraph_smiles is None:
#                 print(f"    No valid subgraph found for idx {idx}, using empty subgraph.")
#                 feature_dim = graphs[idx].x.size(1)
#                 return Data(x=torch.empty((0, feature_dim), dtype=graphs[idx].x.dtype))
#
#             sub_nx = to_nx(subgraph)
#             gm = isomorphism.GraphMatcher(graph_nx, sub_nx)
#
#             if gm.subgraph_is_isomorphic():
#                 break
#             else:
#                 invalid_sub.add(largest_subgraph_smiles)
#
#         node_mappings = generate_node_mappings(graph, subgraph)
#         subgraph.node_mappings = node_mappings
#         return subgraph
#
#     def _precompute_masks(self):
#         """
#         预计算所有图的频繁子图，使用多进程并行优化
#         这样在__getitem__时就不需要重复计算SMILES转换和子图匹配
#         """
#         # Set sharing strategy to file_system to avoid FD limits with file_descriptor
#         torch.multiprocessing.set_sharing_strategy('file_system')
#
#         num_processes = min(64, mp.cpu_count(), len(self.graphs))  # Reduced number of processes
#         print(f"  Using {num_processes} processes for parallel computation")
#
#         # Prepare the partial function using staticmethod
#         process_func = partial(
#             GraphTrainData._compute_single_mask,
#             graphs=self.graphs,
#             pred_labels=self.pred_labels,
#             vocab=self.vocab
#         )
#
#         with mp.Pool(processes=num_processes) as pool:
#             results = list(tqdm(pool.imap(process_func, range(len(self.graphs))), total=len(self.graphs),
#                                 desc="  Computing masks"))
#
#         self.subgraphs = results
#
#         print("  Parallel precomputation completed")
#
#     def _graph_padding(self, num_nodes, graphs):
#         padded_graphs = []
#         for graph in graphs:
#             current_num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
#             new_graph = graph.clone()
#             if current_num_nodes < num_nodes:
#                 padding_size = num_nodes - current_num_nodes
#                 feature_dim = graph.x.size(1)
#                 zero_features = torch.zeros(padding_size, feature_dim, dtype=graph.x.dtype, device=graph.x.device)
#                 new_x = torch.cat([graph.x, zero_features], dim=0)
#                 new_graph.x = new_x
#                 new_graph.num_nodes = num_nodes
#                 mask = torch.cat([
#                     torch.ones(current_num_nodes, dtype=torch.bool, device=graph.x.device),
#                     torch.zeros(padding_size, dtype=torch.bool, device=graph.x.device)
#                 ])
#             else:
#                 mask = torch.ones(current_num_nodes, dtype=torch.bool, device=graph.x.device)
#             new_graph.real_mask = mask
#             padded_graphs.append(new_graph)
#         return padded_graphs
#
#     def __len__(self):
#         return len(self.graphs)
#
#     def __getitem__(self, idx):
#         """
#         获取第idx个图及其预计算的频繁子图掩码
#
#         Returns:
#             dict: {
#                 "graph": PyG Data对象,
#                 "sub_mask": 频繁子图的边掩码 (edge_mask)
#             }
#         """
#         return {
#             "graph": self.graphs[idx],
#             # "sub_mask": self.sub_masks[idx],
#             "subgraph": self.subgraphs[idx]
#         }

class GraphTrainData(Dataset):
    """
    用于训练的图数据集类

    从dataloader中提取图数据，并为每个图找到对应类别的最大频繁子图掩码
    """

    def __init__(self, args, dataloader, gnn, vocab):
        """
        初始化GraphTrainData数据集

        Args:
            dataloader: PyG的DataLoader，用于加载图数据
            gnn: 训练好的GNN分类器，用于预测图的类别
            vocab: 频繁子图的SMILES字典
                   格式: {0: [smiles1, smiles2, ...], 1: [smiles3, smiles4, ...]}
                   key为类别标签，value为该类别的频繁子图SMILES列表
            device: 设备 (cpu or cuda)
        """
        self.gnn = gnn.eval()  # 设置为评估模式
        self.vocab = vocab
        self.device = args.device
        self.graphs = []  # 存储所有单个图
        self.subgraphs = []  # 存储所有图的频繁子图
        self.pred_labels = []  # 存储GNN预测的标签
        self.sub_masks = []  # 预计算的频繁子图掩码
        self.dataset_name = args.dataset
        self.thresh = args.threshold

        # 预处理：从dataloader中提取所有图并进行预测
        print("  Processing graphs and computing subgraph masks...")
        self._process_graphs(dataloader)
        # 预计算所有频繁子图掩码
        print("  Precomputing subgraph masks...")
        self._precompute_masks()
        print(f"  Precomputation completed for {len(self.graphs)} graphs")
        print(f"  length of sub_masks: {len(self.sub_masks)}")
        print(f"  length of graphs: {len(self.graphs)}")
        print(f"  length of subgraphs: {len(self.subgraphs)}")
        #向args添加子图集合中最大原子数
        if args.train_mode:
            args.max_subgraph_nodes = max([g.num_nodes for g in self.subgraphs]) + 3 if self.subgraphs else 0
        self.max_subgraph_nodes = args.max_subgraph_nodes
        print(f"  Max nodes in subgraphs: {args.max_subgraph_nodes}")
        print(f"Padding subgraph with {self.max_subgraph_nodes}")
        self.subgraphs = self._graph_padding(self.max_subgraph_nodes,self.subgraphs)

    def _process_graphs(self, dataloader):
        """从dataloader中提取图并使用GNN进行预测"""
        with torch.no_grad():
            for data in dataloader:
                x = data.x.to(self.device)
                edge_index = data.edge_index.to(self.device)
                batch_idx = data.batch.to(self.device)
                edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None

                # 使用GNN进行预测
                pred_probs, _ = self.gnn.get_pred(x, edge_index, batch_idx)

                # 逐个处理batch中的图
                num_graphs = data.num_graphs
                for i in range(num_graphs):

                    # 计算最大预测概率
                    pred_label = torch.argmax(pred_probs[i]).item()
                    max_prob = pred_probs[i, pred_label].item()

                    # 仅当置信度 >= 0.9 时才添加
                    if max_prob < self.thresh:
                        continue

                    # 提取单个图的数据
                    mask = (batch_idx == i)
                    node_x = x[mask]

                    # 提取单个图的边
                    edge_mask = mask[edge_index[0]]
                    graph_edge_index = edge_index[:, edge_mask]
                    graph_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

                    # 重新映射边索引为局部索引
                    unique_nodes = torch.unique(graph_edge_index)
                    node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}
                    graph_edge_index = torch.tensor(
                        [
                            [node_mapping[old.item()] for old in graph_edge_index[0]],
                            [node_mapping[old.item()] for old in graph_edge_index[1]]
                        ],
                        device=self.device,
                        dtype=torch.long
                    )

                    # 构建单个图的Data对象
                    graph_data = Data(
                        x=node_x,
                        edge_index=graph_edge_index,
                        edge_attr=graph_edge_attr
                    )

                    # 保留原始标签（如果有）
                    if hasattr(data, 'y'):
                        graph_data.y = data.y[i].clone() if data.y.dim() > 0 else data.y.clone()

                    self.graphs.append(graph_data)

                    # 保存预测标签
                    self.pred_labels.append(pred_label)

    def _precompute_masks(self):
        """
        预计算所有图的频繁子图掩码
        这样在__getitem__时就不需要重复计算SMILES转换和子图匹配
        """

        find_largest_subgraph_time = 0.0
        generate_subgraph_mask_time = 0.0
        smarts_to_data_time = 0.0
        for idx in tqdm(range(len(self.graphs)), desc="  Computing masks"):
            graph = self.graphs[idx]
            subgraph = None
            pred_label = self.pred_labels[idx]

            # 转换为SMILES（只需计算一次）
            # graph.smiles = data_to_smiles(graph)

            category_smiles = self.vocab.get(pred_label, [])
            invalid_sub = set()

            # 只转换一次原图
            graph_nx = to_nx(graph)

            while True:
                t1 = time.time()
                subgraph, largest_subgraph_smiles = find_largest_subgraph(self.dataset_name, category_smiles, graph, invalid_sub)
                t2 = time.time()
                find_largest_subgraph_time += (t2 - t1)

                if largest_subgraph_smiles is None:
                    print("    No valid subgraph found, using empty subgraph.")
                    break

                sub_nx = to_nx(subgraph)
                gm = isomorphism.GraphMatcher(graph_nx, sub_nx)

                if gm.subgraph_is_isomorphic():
                    break
                else:
                    invalid_sub.add(largest_subgraph_smiles)
            t1 = time.time()
            node_mappings = generate_node_mappings(graph, subgraph)
            t2 = time.time()
            generate_subgraph_mask_time += (t2 - t1)
            subgraph.node_mappings = torch.tensor(node_mappings)
            self.subgraphs.append(subgraph)

        print(f"    Time spent in find_largest_subgraph: {find_largest_subgraph_time:.4f} seconds")
        print(f"    Time spent in smarts_to_data: {smarts_to_data_time:.4f} seconds")
        print(f"    Time spent in generate_subgraph_mask: {generate_subgraph_mask_time:.4f} seconds")

    def _graph_padding(self, num_nodes, graphs):
        padded_graphs = []
        for graph in graphs:
            current_num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
            new_graph = graph.clone()
            if current_num_nodes < num_nodes:
                padding_size = num_nodes - current_num_nodes
                feature_dim = graph.x.size(1)
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



def custom_collate_fn(batch):
    ori_graphs, tgt_graphs, ori_preds, tgt_preds, ori_embs, tgt_embs, distances = zip(*[
        (item['ori_graph'], item['tgt_graph'], item['ori_pred'], item['tgt_pred'], item['ori_emb'], item['tgt_emb'], item['distance'])
        for item in batch
    ])

    return {
        'ori_graph': Batch.from_data_list(ori_graphs),
        'tgt_graph': Batch.from_data_list(tgt_graphs),
        'ori_pred': torch.tensor(ori_preds),
        'tgt_pred': torch.tensor(tgt_preds),
        'ori_emb': torch.stack(ori_embs),
        'tgt_emb': torch.stack(tgt_embs),
        'distance': torch.tensor(distances)
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

    # 使用Batch.from_data_list合并图
    batched_graphs = Batch.from_data_list(graphs)
    batched_subgraphs = Batch.from_data_list(subgraphs)

    # 返回字典，sub_masks保持为列表
    return {
        'graphs': batched_graphs,
        # 'sub_masks': sub_masks,
        'subgraphs': batched_subgraphs
    }











