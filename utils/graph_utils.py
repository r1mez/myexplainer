import networkx as nx
import numpy as np
import torch
from networkx.classes import nodes
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from typing import Set, Tuple, List, Union
import re
from time import time

MUTAG_atom_map = { 0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
BBBP_atom_map = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 15: 'P', 16: 'S', 17: 'Cl', 20: 'Ca', 35: 'Br', 53: 'I'}
atom_map = {"mutag":MUTAG_atom_map, "bbbp":BBBP_atom_map}

MUTAG_bond_map = {0: Chem.BondType.SINGLE,1: Chem.BondType.DOUBLE,2: Chem.BondType.TRIPLE}
BBBP_bond_map = {1: Chem.BondType.SINGLE,2: Chem.BondType.DOUBLE,3: Chem.BondType.TRIPLE,4: Chem.BondType.AROMATIC}
bond_map = {"mutag":MUTAG_bond_map, "bbbp":BBBP_bond_map}

MUTAG_idx_map = {'C': 0, 'O': 1, 'Cl': 2, 'H': 3, 'N': 4, 'F': 5, 'Br': 6, 'S': 7, 'P': 8, 'I': 9, 'Na': 10, 'K': 11,
                  'Li': 12, 'Ca': 13}

def smarts_to_data(dataset_name, smarts):
    """
    Convert a SMARTS string to a torch_geometric.data.Data object with one-hot encoded atom features
    (including explicit hydrogens) and one-hot encoded edge features based on MUTAG_edge_map,
    with aromatic bonds alternating between single and double.

    Args:
        smarts (str): SMARTS string representing a molecular pattern

    Returns:
        Data: torch_geometric Data object containing molecular graph information
    """
    # Convert SMARTS to RDKit molecule
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Invalid SMARTS string: {smarts}")

    # Add explicit hydrogens
    # mol = Chem.AddHs(mol)

    # Ensure aromaticity is computed
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        # AllChem.ComputeAromaticity(mol)
    except:
        raise ValueError(f"Failed to sanitize or compute aromaticity for SMARTS: {smarts}")

    # Get atom features (one-hot encoding based on MUTAG_atom_map)
    num_atoms = mol.GetNumAtoms()
    num_atom_types = len(atom_map[dataset_name])
    atom_features = []

    for atom in mol.GetAtoms():
        # Get atom symbol
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_map[dataset_name].values():
            raise ValueError(f"Atom type {atom_symbol} not found in {dataset_name}_atom_map")

        # Create one-hot encoding
        one_hot = [0] * num_atom_types
        one_hot[MUTAG_idx_map[atom_symbol]] = 1
        atom_features.append(one_hot)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices and edge attributes
    edge_index = []
    edge_attr = []
    num_edge_types = len(bond_map[dataset_name])  # 3 types: single, double, triple
    # Track bonds to alternate single/double for aromatic bonds
    bond_alternation = {}  # bond_idx -> mapped_bond_type (0 for single, 1 for double)

    # Identify aromatic bonds directly from bond properties
    aromatic_bonds = set()
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            aromatic_bonds.add(bond.GetIdx())

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Determine bond type
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.5 or bond_idx in aromatic_bonds:
            # Aromatic bond: alternate between single (0) and double (1)
            if bond_idx not in bond_alternation:
                # Alternate based on bond index to ensure consistency
                bond_alternation[bond_idx] = 0 if len(bond_alternation) % 2 == 0 else 1
            mapped_bond_type = bond_alternation[bond_idx]
        elif bond_type in [1.0, 2.0, 3.0]:
            # Non-aromatic: single (1.0) -> 0, double (2.0) -> 1, triple (3.0) -> 2
            mapped_bond_type = int(bond_type - 1)
        else:
            raise ValueError(f"Unsupported bond type: {bond_type} for bond {bond_idx}")

        # Create one-hot encoding for edge feature
        edge_feature = [0] * num_edge_types
        edge_feature[mapped_bond_type] = 1

        # Add edges in both directions for undirected graph
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_feature)
        edge_attr.append(edge_feature)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create torch_geometric Data object
    data = Data(
        x=x,  # Node features (one-hot encoded, including hydrogens)
        edge_index=edge_index,  # Edge connectivity (including C-H, N-H bonds)
        edge_attr=edge_attr,  # Edge features (one-hot encoded, based on MUTAG_edge_map)
    )
    # data.idx = 0
    data.smiles = smarts

    return data

def data_to_mol(dataset_name, data):

    # Define bond mapping based on README

    # Create an editable molecule
    mol = Chem.RWMol()

    # Add atoms
    node_to_idx = {}
    for node_idx in range(data.x.size(0)):
        node_feature = data.x[node_idx]
        if dataset_name == "mutag":
            # One-hot encoded node features
            atom_idx = torch.argmax(node_feature).item()
        if dataset_name == "bbbp":
            atom_idx = node_feature[0].item()

        atom_type = atom_map[dataset_name][atom_idx]
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)
        node_to_idx[node_idx] = atom_idx

    # Process edges to remove duplicates (handle undirected graph)

    edge_to_idx = {}
    seen_edges = set()
    for edge_idx in range(data.edge_index.size(1)):
        src, dst = data.edge_index[:, edge_idx].tolist()
        edge = tuple(sorted([src, dst]))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)

        # Handle edge attributes
        edge_feature = data.edge_attr[edge_idx]
        if dataset_name == "mutag":
            # One-hot encoded bonds
            bond_idx = torch.argmax(edge_feature).item()
        if dataset_name == "bbbp":
            # Single integer label
            bond_idx = edge_feature[0].item()

        bond_type = bond_map[dataset_name][bond_idx]
        if src >= 0 and dst >= 0:
            mol.AddBond(src, dst, bond_type)
            # 记录边索引映射
            edge_to_idx[edge_idx] = {
                'src': src,
                'dst': dst,
                'bond_type_idx': bond_idx,
                'bond_type': bond_type
            }
        else:
            print("---------")
            raise ValueError(f"Invalid bond indices: src={src}, dst={dst}")

    mol = mol.GetMol()
    _sanitize_with_valence_correction(mol)  # Use custom sanitization with valence correction
    mol = Chem.RemoveHs(mol, sanitize=False)
    return mol, node_to_idx, edge_to_idx

def smarts_to_mol(smarts):
    return data_to_mol(smarts_to_data(smarts))[0]

def _correct_valence(mol: Chem.RWMol, c_atom: Chem.Atom) -> None:
    """
    Correct valence and charge for a specific atom in the molecule.
    Based on the reference code's logic for handling nitrogen and oxygen atoms.

    Args:
        mol: RDKit molecule (RWMol)
        c_atom: The atom to correct (Chem.Atom)
    """
    neighbors = c_atom.GetNeighbors()
    neighbor_types = [n.GetSymbol() for n in neighbors]

    if c_atom.GetSymbol() == "N":
        # Nitro group (2 or 3 oxygen neighbors)
        if neighbor_types.count("O") > 1:
            for n in neighbors:
                bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
                if (n.GetSymbol() == "O" and
                        bond.GetBondTypeAsDouble() == 1.0 and
                        len(n.GetNeighbors()) == 1):
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)

        # RN(O)=NR or R2[N+]=O
        elif neighbor_types.count("O") == 1:
            for n in neighbors:
                if n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                        c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1:
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)
                elif n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                        c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2:
                    c_atom.SetFormalCharge(1)

        # Diazo group
        elif neighbor_types.count("N") == 1:
            bonds = [b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]
            if 3.0 in bonds:  # C-[N+]#[N]
                c_atom.SetFormalCharge(1)
            elif len(bonds) <= 3:  # Ammonium cation
                c_atom.SetFormalCharge(1)
            else:  # C=[N+]=[N-]
                for n in neighbors:
                    if n.GetSymbol() == "N":
                        c_atom.SetFormalCharge(1)
                        n.SetFormalCharge(-1)

        # Azides
        elif neighbor_types.count("N") > 1:
            for n in neighbors:
                if n.GetSymbol() == "N" and len([nn.GetSymbol() for nn in n.GetNeighbors()]) == 1:
                    c_atom.SetFormalCharge(1)
                    n.SetFormalCharge(-1)
                else:
                    c_atom.SetFormalCharge(1)

        # Ammonium cations
        elif neighbor_types.count("C") >= 3:
            c_atom.SetFormalCharge(1)

        else:
            raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")

    elif c_atom.GetSymbol() == "O":
        # Oxygen with single and double bonds
        if set([b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]) == {1.0, 2.0}:
            c_atom.SetFormalCharge(1)
        else:
            raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")

    else:
        raise ValueError(f"Unexpected atom {c_atom.GetSymbol()} with idx {c_atom.GetIdx()}")


def _sanitize_with_valence_correction(mol: Chem.RWMol) -> None:
    """
    Sanitize molecule and correct valence/charge if needed.

    Args:
        mol: RDKit molecule (RWMol)

    Raises:
        ValueError: If sanitization or valence correction fails
    """
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.AtomValenceException as e:
        match = re.search(r"atom # (\d+)", str(e))
        if match:
            atom_idx = int(match.group(1))
            _correct_valence(mol, mol.GetAtomWithIdx(atom_idx))
            _sanitize_with_valence_correction(mol)  # Recursive call after correction
        else:
            raise ValueError(f"No atom number in exception: {str(e)}")


def extract_explanatory_subgraph(
    original: Union[Data, Batch, List[Data]],
    counterfactual: Union[Data, Batch, List[Data]]
) -> Union[Data, Batch, List[Data]]:
    """
    解释子图：只基于边集变化（无向意义上的边集差异）。

    规则：
    - 解释边 =
        * 原图有但反事实没有的边
        * 反事实有但原图没有的边
      （用无向 canonical 形式比较：{min(u,v), max(u,v)}）
    - 解释子图的节点特征来自原图。
    - 如果原图和反事实图在 canonical 边集上一样：
        -> 返回的解释子图保留原图所有节点（x不变），但 edge_index 为空。
    - 有变化时：
        -> 只保留“涉及变化边”的端点节点，然后重新编号为紧凑的 0..N-1。
    """

    def _process_pair(orig: Data, cf: Data) -> Data:
        if orig.num_nodes != cf.num_nodes:
            raise ValueError(f"Graph pair mismatch: {orig.num_nodes} vs {cf.num_nodes} nodes.")

        device = orig.x.device
        num_nodes, feat_dim = orig.x.shape

        # 无向 canonical 边集：用于判断边是否存在（忽略方向/多重边）
        def canonical_edges(edge_idx: torch.Tensor) -> set:
            if edge_idx.numel() == 0:
                return set()
            src, tgt = edge_idx
            mins = torch.min(src, tgt)
            maxs = torch.max(src, tgt)
            return {(int(mn), int(mx)) for mn, mx in zip(mins, maxs)}

        orig_canon = canonical_edges(orig.edge_index)
        cf_canon = canonical_edges(cf.edge_index)

        # 如果无向边集完全相同：解释子图 = 所有原图节点 + 空边集
        if orig_canon == cf_canon:
            empty_eidx = torch.empty((2, 0), dtype=torch.long, device=device)
            # 保留原图节点特征
            return Data(x=orig.x.clone(), edge_index=empty_eidx)

        explain_edges = []

        # 1) 原图有、反事实没有的边（保留原图方向）
        src_o, tgt_o = orig.edge_index
        for i in range(orig.edge_index.size(1)):
            u = int(src_o[i])
            v = int(tgt_o[i])
            key = (min(u, v), max(u, v))
            if key not in cf_canon:
                explain_edges.append(orig.edge_index[:, i:i + 1])

        # 2) 反事实有、原图没有的边（保留反事实方向）
        src_c, tgt_c = cf.edge_index
        for i in range(cf.edge_index.size(1)):
            u = int(src_c[i])
            v = int(tgt_c[i])
            key = (min(u, v), max(u, v))
            if key not in orig_canon:
                # 注意把边搬到 orig 所在 device
                explain_edges.append(cf.edge_index[:, i:i + 1].to(device=device))

        # 理论上 orig_canon != cf_canon 时 explain_edges 一定非空，但做个兜底
        if explain_edges:
            explain_eidx = torch.cat(explain_edges, dim=1)
        else:
            explain_eidx = torch.empty((2, 0), dtype=torch.long, device=device)

        # 从解释边的端点收集需要保留的节点
        nodes_set = set()
        if explain_eidx.numel() > 0:
            nodes_set.update(int(u) for u in explain_eidx[0])
            nodes_set.update(int(v) for v in explain_eidx[1])

        # 极端兜底：如果 nodes_set 竟然为空，就退化成“保留所有节点、无边”
        if not nodes_set:
            empty_eidx = torch.empty((2, 0), dtype=torch.long, device=device)
            return Data(x=orig.x.clone(), edge_index=empty_eidx)

        node_list = sorted(nodes_set)
        node_map = {old: new for new, old in enumerate(node_list)}

        # 重新取节点特征（来自原图）
        idx_tensor = torch.tensor(node_list, dtype=torch.long, device=device)
        explain_x = orig.x[idx_tensor]

        # 重新编号解释边
        new_src = torch.tensor(
            [node_map[int(u)] for u in explain_eidx[0]],
            dtype=torch.long,
            device=device,
        )
        new_tgt = torch.tensor(
            [node_map[int(v)] for v in explain_eidx[1]],
            dtype=torch.long,
            device=device,
        )
        new_eidx = torch.stack([new_src, new_tgt], dim=0)

        return Data(x=explain_x, edge_index=new_eidx)

    # 输入归一化：统一成 List[Data]
    def normalize_input(inp: Union[Data, Batch, List[Data]]) -> List[Data]:
        if isinstance(inp, Data):
            return [inp]
        elif isinstance(inp, Batch):
            return inp.to_data_list()
        elif isinstance(inp, list):
            if not all(isinstance(g, Data) for g in inp):
                raise ValueError("List must contain Data objects.")
            return inp
        else:
            raise ValueError("Input must be Data, Batch, or list of Data.")

    orig_list = normalize_input(original)
    cf_list = normalize_input(counterfactual)

    if len(orig_list) != len(cf_list):
        raise ValueError("Original and counterfactual inputs must have matching number of graphs.")

    explain_list = [_process_pair(o, c) for o, c in zip(orig_list, cf_list)]

    # 返回类型与输入保持一致
    if len(explain_list) == 1:
        return explain_list[0]
    elif isinstance(original, Batch) or isinstance(counterfactual, Batch):
        return Batch.from_data_list(explain_list)
    else:
        return explain_list


def exclude_explanatory_subgraph(original: Union[Data, Batch, List[Data]],
                                 counterfactual: Union[Data, Batch, List[Data]]) -> Union[Data, Batch, List[Data]]:
    """
    Efficient batch-enabled non-explanatory subgraph extraction (excludes explanatory parts).

    Supports single Data, lists of Data, or Batch objects for both inputs.
    Processes each graph pair independently using vectorized operations where possible.
    For Batch inputs, uses to_data_list() for per-graph processing, then reconstructs Batch.
    Time complexity: O(sum(V + E) over all graphs), suitable for batched inference.

    Logic (vectorized where feasible):
    - Changed nodes: Vectorized mask via (original.x != counterfactual.x).any(-1).
    - Retained edges: From original, only if exists in counterfactual AND not both endpoints changed.
    - Node features: Original, but zeroed for changed nodes.
    - All nodes preserved; only edges filtered.
    """

    def _process_pair(orig: Data, cf: Data) -> Data:
        if orig.num_nodes != cf.num_nodes:
            raise ValueError(f"Graph pair mismatch: {orig.num_nodes} vs {cf.num_nodes} nodes.")

        device = orig.x.device
        num_nodes, feat_dim = orig.x.shape

        # Vectorized changed nodes mask
        changed_mask = ~(orig.x == cf.x).all(-1)  # bool tensor [num_nodes]
        changed_nodes = torch.nonzero(changed_mask).flatten()  # tensor [num_changed]
        changed_set = set(changed_nodes.tolist())

        # Canonical edge sets: sorted (min(u,v), max(u,v)) as tuples in sets
        def canonical_edges(edge_idx: torch.Tensor) -> set:
            src, tgt = edge_idx
            mins = torch.min(src, tgt)
            maxs = torch.max(src, tgt)
            return {(min.item(), max.item()) for min, max in zip(mins, maxs)}

        orig_canon = canonical_edges(orig.edge_index)
        cf_canon = canonical_edges(cf.edge_index)

        # Collect keep edge indices (keep original direction)
        keep_edges = []  # list of [2,1] tensors

        src_o, tgt_o = orig.edge_index
        for i in range(orig.num_edges):
            u, v = src_o[i].item(), tgt_o[i].item()
            edge_key = (min(u, v), max(u, v))

            # Rule: Skip if not in counterfactual
            if edge_key not in cf_canon:
                continue

            # Rule: Skip if both endpoints changed
            if u in changed_set and v in changed_set:
                continue

            # Retain
            keep_edges.append(orig.edge_index[:, i:i + 1])

        # Concat edges
        if keep_edges:
            keep_eidx = torch.cat(keep_edges, dim=1)
        else:
            keep_eidx = torch.empty((2, 0), dtype=torch.long, device=device)

        # Node features: original, zero changed
        non_explain_x = orig.x.clone()
        if len(changed_nodes) > 0:
            non_explain_x[changed_nodes] = 0

        return Data(x=non_explain_x, edge_index=keep_eidx)

    # Input normalization
    def normalize_input(inp: Union[Data, Batch, List[Data]]) -> List[Data]:
        if isinstance(inp, Data):
            return [inp]
        elif isinstance(inp, Batch):
            return inp.to_data_list()
        elif isinstance(inp, list):
            if not all(isinstance(g, Data) for g in inp):
                raise ValueError("List must contain Data objects.")
            return inp
        else:
            raise ValueError("Input must be Data, Batch, or list of Data.")

    orig_list = normalize_input(original)
    cf_list = normalize_input(counterfactual)

    if len(orig_list) != len(cf_list):
        raise ValueError("Original and counterfactual inputs must have matching number of graphs.")

    non_explain_list = [_process_pair(o, c) for o, c in zip(orig_list, cf_list)]

    if len(non_explain_list) == 1:
        return non_explain_list[0]
    elif isinstance(original, Batch) or isinstance(counterfactual, Batch):
        return Batch.from_data_list(non_explain_list)
    else:
        return non_explain_list


def process_outputs(args, outputs):
    """
    将重构的节点特征 x_recon 和邻接矩阵 adj_recon 转换为 PyG 格式的 Batch。

    参数:
        args: 包含配置信息的参数对象
            - max_subgraph_nodes: 最大节点数
            - x_dim: 节点特征维度
            - device: 设备
        outputs: dict，模型输出，包含
            - 'x_recon': 重构的节点特征，shape [B, N*F] 或 [B, N, F]
            - 'adj_recon': 重构的邻接矩阵，shape [B, N*N] 或 [B, N, N]

    返回:
        Batch: PyG的Batch对象，包含batch_size个重构图
    """
    device = args.device
    max_num_nodes = args.max_subgraph_nodes
    x_dim = args.x_dim

    # 从outputs中提取重构结果
    x_recon = outputs['x_recon']  # [B, N*F] 或 [B, N, F]
    adj_recon = outputs['adj_recon']  # [B, N*N] 或 [B, N, N]

    # 1. 确保形状正确：将扁平化的张量reshape为 (batch_size, max_num_nodes, *)
    if x_recon.dim() == 2:
        # [B, N*F] -> [B, N, F]
        batch_size = x_recon.shape[0]
        x_recon = x_recon.view(batch_size, max_num_nodes, x_dim)
    else:
        # 已经是 [B, N, F]
        batch_size = x_recon.shape[0]

    if adj_recon.dim() == 2:
        # [B, N*N] -> [B, N, N]
        adj_recon = adj_recon.view(batch_size, max_num_nodes, max_num_nodes)
    # 否则已经是 [B, N, N]

    # 2. 为每个batch中的图创建PyG Data对象
    graphs = []
    for i in range(batch_size):
        x_i = x_recon[i]  # [N, F]
        adj_i = adj_recon[i]  # [N, N]

        # 3. 将邻接矩阵转换为edge_index
        # 设置阈值，将概率值转换为0/1（可根据需要调整阈值）
        threshold = 0.5
        mask = adj_i > threshold

        # 获取所有边的索引 [2, E]
        edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()

        # 4. 创建PyG Data对象
        graph = Data(
            x=x_i,  # [N, F]
            edge_index=edge_index,  # [2, E]
            num_nodes=max_num_nodes
        )
        graphs.append(graph)

    # 5. 将所有图合并为一个Batch
    batch = Batch.from_data_list(graphs).to(device)

    return batch
