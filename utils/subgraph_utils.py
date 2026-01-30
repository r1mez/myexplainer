from typing import Optional

import networkx as nx
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.utils import to_networkx, dense_to_sparse

from joblib import Parallel, delayed
import multiprocessing

RDLogger.DisableLog('rdApp.*')  # 关闭所有 rdApp 相关的警告，包括 valence、SMARTS 和 Kekulization
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
import torch
import numpy as np
import re

def generate_subgraph_mask(
    sub_smiles: str,
    data: Data,
):
    """
    根据给定的 SMILES 和 Data 对象，生成子图的节点和边掩码，以及节点映射

    参数：
        sub_smiles: 子图的 SMILES 字符串
        data: torch_geometric.data.Data 对象，其中成员的 x 和 edge_attr 为 one-hot 编码
        atom_map: 原子类型映射，键为原子符号，值为 one-hot 索引
        bond_map: 键类型映射，键为 one-hot 索引，值为 RDKit 键类型（Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE）

    返回：
        node_mask: 形状为 (num_nodes,) 的布尔张量，表示子图中的节点
        edge_mask: 形状为 (num_edges,) 的布尔张量，表示子图中的边
        node_mappings: list[int]，子图节点到原图节点的索引映射
        如果匹配失败，返回 (None, None, None)
    """
    try:
        # Convert sub_smiles to RDKit Mol
        sub_mol = Chem.MolFromSmiles(sub_smiles)
        if sub_mol is None:
            return None, None, None

        # Convert data to RDKit Mol using provided tool function
        # Note: atom_map and bond_map are used internally in data_to_mol, but since it's hardcoded, we proceed
        mol, _, _ = data_to_mol(data)

        # Check for substructure match
        if not mol.HasSubstructMatch(sub_mol):
            return None, None, None

        # Get the atom mapping from substructure to structure
        match = mol.GetSubstructMatch(sub_mol)
        if not match:
            return None, None, None

        # node_mappings is the list of matched atom indices in the original molecule
        node_mappings = list(match)

        matched_atoms = set(node_mappings)

        # Create node_mask
        num_nodes = data.num_nodes
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for i in matched_atoms:
            node_mask[i] = True

        # Create edge_mask
        num_edges = data.num_edges
        edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        for e in range(num_edges):
            u, v = data.edge_index[:, e].tolist()
            if u in matched_atoms and v in matched_atoms:
                edge_mask[e] = True

        return node_mask, edge_mask, node_mappings

    except Exception as e:
        return None, None, None


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


def concat_graphs(args, outputs, batch):
    """
    将MyExplainer输出的重构子图特征和邻接矩阵替换原图中对应的诱导子图部分

    参数:
        args: 参数对象，包含device, max_subgraph_nodes等
        outputs: MyExplainer模型的输出字典，包含:
            - 'x_recon': 重构的节点特征 (batch_size, max_subgraph_nodes * x_dim)
            - 'adj_recon': 重构的邻接矩阵 (batch_size, max_subgraph_nodes * max_subgraph_nodes)
        batch: 批量数据字典，包含:
            - 'graphs': 原始图的Batch对象
                - 'x': 原始图的节点特征 (one-hot编码)
                - 'edge_index': 原始图的边索引
            - 'subgraphs': 诱导子图的Batch对象
                - 'x': 诱导子图的节点特征 (one-hot编码)
                - 'edge_index': 诱导子图的边索引
                - 'node_mappings': 每个tensor包含子图节点到原图节点的索引映射
                - 'real_mask': 布尔tensor，表示子图中哪些节点是真实节点（非padding）

    返回:
        reconstructed_graphs: list of dict，每个dict包含:
            - 'x': 重构后完整图的节点特征 (one-hot编码)
            - 'edge_index': 重构后完整图的边索引
            - 'num_nodes': 节点数量
    """
    device = args.device
    max_subgraph_nodes = args.max_subgraph_nodes

    orig_graphs = batch['graphs'].to_data_list()
    sub_graphs = batch['subgraphs'].to_data_list()
    batch_size = len(orig_graphs)

    x_dim = orig_graphs[0].x.size(1)
    x_recon = outputs['x_recon'].view(batch_size, max_subgraph_nodes, x_dim)
    adj_recon = outputs['adj_recon'].view(batch_size, max_subgraph_nodes, max_subgraph_nodes)
    # adj_recon = torch.zeros(batch_size, max_subgraph_nodes, max_subgraph_nodes, device=outputs['adj_recon'].device,
    #                         dtype=outputs['adj_recon'].dtype)

    reconstructed_graphs = []

    for i in range(batch_size):
        orig = orig_graphs[i]                           # 原图
        sub = sub_graphs[i]                             # 取出的频繁子图
        mapping = sub.node_mappings.to(device)          # 子图 - > 原图的依次节点映射
        real_mask = sub.real_mask.to(device)
        real_indices = torch.where(real_mask)[0]        # 子图中除去填充节点的真实节点索引
        real_n = real_indices.size(0)                   # 子图中除去填充节点的真实节点数

        # 重构节点特征
        output_x = orig.x.clone()                        # 原图节点特征矩阵（ori_num_nodes * x_dim），准备根据模型输出重构节点特征
        if real_n > 0:

            if args.dataset == 'ba2motif':
                x_recon_sub = x_recon[i, real_indices]       # 子图中真实节点部分的重构特征 (real_n, x_dim)
                output_x[mapping] = x_recon_sub
            else:
                x_recon_sub = x_recon[i, real_indices]       # 子图中真实节点部分的重构特征 (real_n, x_dim)

                # Straight-Through Estimator: 前向用离散值，反向传连续梯度
                atom_indices = torch.argmax(x_recon_sub, dim=-1)  # (real_n,)
                x_recon_onehot = torch.zeros_like(x_recon_sub)  # (real_n, x_dim)
                x_recon_onehot.scatter_(1, atom_indices.unsqueeze(-1), 1.0)

                if args.train_mode:
                    # 训练模式：使用 STE
                    # 前向：one-hot 离散值
                    # 反向：梯度绕过 argmax，直接传给 x_recon_sub
                    x_recon_final = x_recon_onehot - x_recon_sub.detach() + x_recon_sub
                else:
                    # 评估模式：纯离散化
                    x_recon_final = x_recon_onehot

                # 赋值到原图
                output_x[mapping] = x_recon_final


        # Reconstruct edge_index
        sub_nodes_mask = torch.zeros(orig.num_nodes, dtype=torch.bool, device=device)
        if real_n > 0:
            sub_nodes_mask[mapping] = True

        # Identify and remove original within-subgraph edges
        within_mask = sub_nodes_mask[orig.edge_index[0]] & sub_nodes_mask[orig.edge_index[1]]
        keep_mask = ~within_mask
        recon_edge_index = orig.edge_index[:, keep_mask]

        # Add reconstructed within-subgraph edges
        if real_n > 0:
            adj_sub = adj_recon[i, real_indices[:, None], real_indices[None, :]]  # (real_n, real_n)
            adj_hard = (adj_sub > 0.5).float()
            if args.train_mode:
                # 训练模式：使用 STE
                # 前向：adj_hard 离散值 (下游 dense_to_sparse 生成离散边)
                # 反向：梯度绕过阈值化，直接传给 adj_sub
                adj_final = adj_hard - adj_sub.detach() + adj_sub
            else:
                # 评估模式：纯离散化 (无梯度)
                adj_final = adj_hard
            sub_edge_index_sub, _ = dense_to_sparse(adj_final)
            if sub_edge_index_sub.size(1) > 0:
                sub_edge_index_full = torch.stack([
                    mapping[sub_edge_index_sub[0]],
                    mapping[sub_edge_index_sub[1]]
                ], dim=0)
                recon_edge_index = torch.cat([recon_edge_index, sub_edge_index_full], dim=1)

        recon_data = Data(
            x=output_x,
            edge_index=recon_edge_index,
            # batch = batch['graphs'].batch
        )
        reconstructed_graphs.append(recon_data)

    reconstructed_graphs = Batch.from_data_list(reconstructed_graphs)

    return reconstructed_graphs


def _correct_valence(mol: Chem.RWMol, c_atom: Chem.Atom) -> None:
    neighbors = c_atom.GetNeighbors()
    neighbor_types = [n.GetSymbol() for n in neighbors]

    if c_atom.GetSymbol() == "N":
        if neighbor_types.count("O") > 1:
            for n in neighbors:
                bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
                if (n.GetSymbol() == "O" and
                        bond.GetBondTypeAsDouble() == 1.0 and
                        len(n.GetNeighbors()) == 1):
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)
        elif neighbor_types.count("O") == 1:
            for n in neighbors:
                if n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                        c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1:
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)
                elif n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                        c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2:
                    c_atom.SetFormalCharge(1)
        elif neighbor_types.count("N") == 1:
            bonds = [b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]
            if 3.0 in bonds:
                c_atom.SetFormalCharge(1)
            elif len(bonds) <= 3:
                c_atom.SetFormalCharge(1)
            else:
                for n in neighbors:
                    if n.GetSymbol() == "N":
                        c_atom.SetFormalCharge(1)
                        n.SetFormalCharge(-1)
        elif neighbor_types.count("N") > 1:
            for n in neighbors:
                if n.GetSymbol() == "N" and len([nn.GetSymbol() for nn in n.GetNeighbors()]) == 1:
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)
                else:
                    c_atom.SetFormalCharge(1)
        elif neighbor_types.count("C") >= 3:
            c_atom.SetFormalCharge(1)
        else:
            raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")
    elif c_atom.GetSymbol() == "O":
        if set([b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]) == {1.0, 2.0}:
            c_atom.SetFormalCharge(1)
        else:
            raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")
    else:
        raise ValueError(f"Unexpected atom {c_atom.GetSymbol()} with idx {c_atom.GetIdx()}")


def _sanitize_with_valence_correction(mol: Chem.RWMol) -> None:
    import re
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.AtomValenceException as e:
        match = re.search(r"atom # (\d+)", str(e))
        if match:
            atom_idx = int(match.group(1))
            _correct_valence(mol, mol.GetAtomWithIdx(atom_idx))
            _sanitize_with_valence_correction(mol)
        else:
            raise ValueError(f"No atom number in exception: {str(e)}")




