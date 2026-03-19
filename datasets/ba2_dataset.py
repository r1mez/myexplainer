## In[Import]
import random
import numpy as np
import os.path as op
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


## In[Generate Dataset with exp_gt]
# def generate_ba2motif_with_exp_gt(num_graphs=1000, save_path='ba2motif_with_exp_gt'):
#     adj_list, x_list, y_list, exp_gt_list = [], [], [], []
#     for i in range(num_graphs):
#         # 生成 BA 基底图
#         G = nx.barabasi_albert_graph(20, 2)  # n=20, m=2
#         y = random.choice([0, 1])  # 0: House, 1: Cycle
#         motif_nodes_start = 20  # 模体节点从 20 开始，避免重叠
#
#         if y == 0:  # House 结构
#             cycle_nodes = list(range(motif_nodes_start, motif_nodes_start + 5))  # [20,21,22,23,24]
#             G.add_nodes_from(cycle_nodes)
#             # G.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 20)])  # 5 节点
#             G.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 20), (24, 20), (24, 23)])  # 5 节点
#             house_node = motif_nodes_start + 4  # 24
#             G.add_node(house_node) #24
#             G.add_edges_from([(house_node, 20), (house_node, 23)])  # 连接到非邻接节点
#             exp_gt = np.array([(20, 21), (21, 22), (22, 23), (23, 20), (house_node, 20), (house_node, 23)])  # 正样本边对
#         else:  # Cycle 结构 (5 节点环)
#             cycle_nodes = list(range(motif_nodes_start, motif_nodes_start + 5))  # [20,21,22,23,24]
#             G.add_nodes_from(cycle_nodes)
#             G.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])
#             exp_gt = np.array([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])  # 正样本边对
#
#         # 附加模体到 BA 图：随机连接一个模体节点到 BA 节点
#         attach_node_ba = random.choice(list(G.nodes())[:20])  # 从 BA 节点选一个
#         attach_node_motif = cycle_nodes[0]  # 模体第一个节点
#         G.add_edge(attach_node_ba, attach_node_motif)
#
#         # 生成 adj, x, y
#         adj = nx.to_numpy_array(G)
#         x = np.ones((adj.shape[0], 10))  # 虚拟特征
#         y_one_hot = np.zeros(2)
#         y_one_hot[y] = 1
#
#         adj_list.append(adj)
#         x_list.append(x)
#         y_list.append(y_one_hot)
#         exp_gt_list.append(exp_gt)
#
#     # 保存
#     np.savez(save_path, adj=adj_list, x=x_list, y=y_list, exp_gt=exp_gt_list)
#     print(f"Saved to {save_path}")
#
#
# ## In[BA2Motif]
# class BA2Motif(InMemoryDataset):
#     splits = ['train', 'valid', 'test']
#
#     def __init__(self, root, mode, transform=None, pre_transform=None, pre_filter=None):
#         assert mode in self.splits
#         self.mode = mode
#         super().__init__(root, transform, pre_transform, pre_filter)
#         idx = self.processed_file_names.index('{}.pt'.format(mode))
#         self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
#
#     @property
#     def raw_file_names(self):
#         return 'ba2motif_with_exp_gt.npz'  # 使用包含 exp_gt 的数据集
#
#     @property
#     def processed_file_names(self):
#         return ['train.pt', 'valid.pt', 'test.pt']
#
#     def download(self):
#         file = self.raw_file_names
#         if not op.exists(op.join(self.raw_dir, file)):
#             print(f'Data {file} does not exist, generating...')
#             generate_ba2motif_with_exp_gt(save_path=op.join(self.raw_dir, file))
#
#     def process(self):
#         # 尝试加载包含 exp_gt 的数据
#         try:
#             data = np.load(op.join(self.raw_dir, self.raw_file_names), allow_pickle=True)
#             adj, x, y, exp_gt = data['adj'], data['x'], data['y'], data['exp_gt']
#         except (KeyError, ValueError):
#             # 如果 exp_gt 缺失，抛出错误提示重新生成
#             print(
#                 f"Error: 'exp_gt' not found in {self.raw_file_names}. Please run generate_ba2motif_with_exp_gt() to create a new dataset.")
#             raise
#
#         graph_list = []
#
#         for i, (adj_i, x_i, y_i, exp_gt_i) in enumerate(zip(adj, x, y, exp_gt)):
#             edge_index = dense_to_sparse(torch.tensor(adj_i, dtype=torch.float))[0]
#             node_index = torch.unique(edge_index)
#             assert node_index.max() == node_index.size(0) - 1, f"Graph {i} has invalid node indices"
#
#             x_tensor = torch.tensor(x_i, dtype=torch.float)
#             y_tensor = torch.tensor(np.argmax(y_i), dtype=torch.long)  # 转换为标量标签
#
#             # 处理 exp_gt（正样本边对）
#             if isinstance(exp_gt_i, np.ndarray) and exp_gt_i.size > 0:
#                 exp_gt_tensor = torch.tensor(exp_gt_i, dtype=torch.long)
#             else:
#                 print(f"Warning: Invalid or empty exp_gt for graph {i}: {exp_gt_i}, setting to None")
#                 exp_gt_tensor = None
#
#             graph = Data(x=x_tensor,
#                          edge_index=edge_index,
#                          y=y_tensor,
#                          exp_gt=exp_gt_tensor)  # 添加 ground-truth 边对
#
#             if self.pre_filter is not None:
#                 graph = self.pre_filter(graph)
#             if self.pre_transform is not None:
#                 graph = self.pre_transform(graph)
#
#             graph_list.append(graph)
#
#         random.shuffle(graph_list)
#         torch.save(self.collate(graph_list[0:400]), self.processed_paths[0])  # Train
#         torch.save(self.collate(graph_list[400:800]), self.processed_paths[1])  # Valid
#         torch.save(self.collate(graph_list[800:]), self.processed_paths[2])  # Test

# if __name__ == '__main__':
#     # 运行生成数据集（仅在需要时运行）
#     generate_ba2motif_with_exp_gt()


class BA2Motif(InMemoryDataset):
    splits = ['training', 'evaluation', 'testing']

    def __init__(self, root, mode,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode

        super().__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return 'ba2motif.pkl'

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):
        file = 'ba2motif.pkl'
        # print('self.raw_dir:', self.raw_dir)
        # print('op.join(self.raw_dir, file):',op.join(self.raw_dir, file))
        if not op.exists(op.join(self.raw_dir, file)):
            print('Data does not exist.')
            raise FileNotFoundError

    def process(self):
        # print('op.join(self.raw_dir,self.raw_file_names):',op.join(self.raw_dir,self.raw_file_names))

        adj, x, y = np.load(op.join(self.raw_dir,
                                    self.raw_file_names),
                            allow_pickle=True)

        graph_list = []

        for i, (adj, x, y) in enumerate(zip(adj, x, y)):
            edge_index = dense_to_sparse(torch.tensor(adj))[0]
            node_index = torch.unique(edge_index)
            assert node_index.max() == node_index.size(0) - 1

            x = torch.tensor(x, dtype=torch.float)
            y = np.argmax(y)
            y = torch.tensor(y, dtype=torch.long)

            # exp_gt = ((edge_index[0] >= 20) & (edge_index[1] >= 20))
            # exp_gt = torch.tensor(exp_gt, dtype = torch.long)

            # 生成 exp_gt，形状为 [num_expert_edges, 2]，仅包含无向的专家边
            mask = ((edge_index[0] >= 20) & (edge_index[1] >= 20))  # 专家边：两个端点 >= 20
            expert_edges = edge_index.t()[mask]  # 提取专家边，形状 [num_expert_edges, 2]

            # 去除重复边（无向图中 (u, v) 和 (v, u) 只保留一个，选 u < v）
            if expert_edges.size(0) > 0:
                # 确保 u < v
                expert_edges = torch.sort(expert_edges, dim=1)[0]  # 按行排序，[min(u,v), max(u,v)]
                # 去重：转换为 tuple 集合再转回 tensor
                expert_edges = torch.unique(expert_edges, dim=0)
            # else:
            #     # 如果没有专家边，返回空张量
            #     expert_edges = torch.tensor([], dtype=torch.long).reshape(0, 2)

            exp_gt = expert_edges  # 形状 [num_expert_edges, 2]

            graph = Data(x=x,
                         edge_index=edge_index,
                         y=y,
                         exp_gt=exp_gt)

            if self.pre_filter is not None:
                graph = self.pre_filter(graph)

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            graph_list.append(graph)

        random.shuffle(graph_list)

        torch.save(self.collate(graph_list[0:700]),
                   self.processed_paths[0])
        torch.save(self.collate(graph_list[700:900]),
                   self.processed_paths[1])
        torch.save(self.collate(graph_list[900:]),
                   self.processed_paths[2])
