from typing import Optional

import networkx as nx
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.utils import to_networkx, dense_to_sparse

RDLogger.DisableLog('rdApp.*')  # 关闭所有 rdApp 相关的警告，包括 valence、SMARTS 和 Kekulization
from torch_geometric.data import Data, Batch
import torch



def to_nx(data: Data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().cpu().numpy().tolist()
    g.add_edges_from(edges)
    for i in range(data.num_nodes):
        try:
            g.nodes[i]['feature'] = torch.argmax(data.x[i]).item()
        except:
            # 如果没有feature，则跳过
             continue
    return g


def generate_node_mappings(
        graph: Data,
        subgraph: Data
) -> Optional[list[int]]:
    """
        生成子图节点到原图节点的索引映射

        参数：
            graph: torch_geometric.data.Data 对象，表示原始图，graph.x为节点特征（以one-hot形式存储），graph.edge_index为边列表
            subgraph: torch_geometric.data.Data 对象，表示子图, 默认为graph的诱导子图，subgraph.x为节点特征（以one-hot形式存储），subgraph.edge_index为边列表
        返回：
            node_mappings: list[int]，子图节点到原图节点的索引
            如果映射失败，返回 None
        """
    if subgraph.num_nodes > graph.num_nodes:
        return None

    graph_nx = to_nx(graph)
    sub_nx = to_nx(subgraph)

    gm = isomorphism.GraphMatcher(graph_nx, sub_nx)

    if gm.subgraph_is_isomorphic():
        for mapping in gm.subgraph_isomorphisms_iter():
            mapping = {v: k for k, v in mapping.items()}
            node_mappings = [mapping[i] for i in range(subgraph.num_nodes)]
            return node_mappings

    return None




