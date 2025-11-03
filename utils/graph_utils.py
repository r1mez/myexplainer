import networkx as nx
import numpy as np
import torch
from networkx.classes import nodes
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Set, Tuple, List
from data.mutag.smiles import data_to_smiles
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


def extract_explanatory_subgraph(original, counterfactual):
    """
        Converts original and counterfactual graphs (both torch_geometric.data.Data)
        into an explanation subgraph based on the specified logic.

        Logic:
        - Traverse original graph's edges: if an edge exists in both, exclude it from explanation.
          If unique to original, include it and its endpoints.
        - For nodes: include if features differ between original and counterfactual.
        - Finally, add edges unique to counterfactual.

        Assumes undirected graphs where edge_index may not be sorted, but canonicalizes edges as (min(u,v), max(u,v)) for existence checks.
        Node features (x) in the output use the original graph's values.
        """
    if original.x.size(0) != counterfactual.x.size(0):
        raise ValueError("Original and counterfactual graphs must have the same number of nodes.")

    num_nodes = original.x.size(0)

    # Get canonical edge sets for existence checks (undirected: (min(u,v), max(u,v)))
    def get_canonical_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
        edge_set = set()
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            edge_set.add((min(u, v), max(u, v)))
        return edge_set

    original_edge_set = get_canonical_edge_set(original.edge_index)
    counterfactual_edge_set = get_canonical_edge_set(counterfactual.edge_index)

    # First, collect changed nodes (node feature differences)
    changed_nodes: Set[int] = set()
    for i in range(num_nodes):
        if not torch.equal(original.x[i], counterfactual.x[i]):
            changed_nodes.add(i)

    # Collect edges unique to original (keep original direction)
    explain_edge_indices: List[torch.Tensor] = []
    for i in range(original.edge_index.size(1)):
        u = original.edge_index[0, i].item()
        v = original.edge_index[1, i].item()
        key = (min(u, v), max(u, v))
        if key not in counterfactual_edge_set:
            explain_edge_indices.append(original.edge_index[:, i:i + 1])

    # Collect edges unique to counterfactual (keep counterfactual direction)
    # for i in range(counterfactual.edge_index.size(1)):
    #     u = counterfactual.edge_index[0, i].item()
    #     v = counterfactual.edge_index[1, i].item()
    #     key = (min(u, v), max(u, v))
    #     if key not in original_edge_set:
    #         explain_edge_indices.append(counterfactual.edge_index[:, i:i + 1])

    # Additional rule: Add edges from original that connect two changed nodes
    for i in range(original.edge_index.size(1)):
        u = original.edge_index[0, i].item()
        v = original.edge_index[1, i].item()
        if u in changed_nodes and v in changed_nodes:
            # Add the edge (even if it exists in counterfactual, as per rule)
            explain_edge_indices.append(original.edge_index[:, i:i + 1])

    # Concatenate all explain edges
    if explain_edge_indices:
        explain_edge_index = torch.cat(explain_edge_indices, dim=1)
    else:
        explain_edge_index = torch.empty((2, 0), dtype=torch.long, device=original.edge_index.device)

    # Collect all relevant nodes: changed + endpoints of explain edges
    all_nodes: Set[int] = changed_nodes.copy()
    for i in range(explain_edge_index.size(1)):
        all_nodes.add(explain_edge_index[0, i].item())
        all_nodes.add(explain_edge_index[1, i].item())

    if not all_nodes:
        # Empty explanation graph - all nodes belong to non-explanation graph
        empty_explain_graph = Data(
            x=torch.empty((0, original.x.size(1)), dtype=original.x.dtype, device=original.x.device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=original.edge_index.device)
        )
        # Non-explanation graph is the entire original graph
        non_explain_graph = Data(x=original.x, edge_index=original.edge_index)
        return empty_explain_graph, non_explain_graph

        # # Retain the first node (index 0) as a fallback to avoid empty graph
        # all_nodes.add(0)

    node_list = sorted(all_nodes)
    num_explain_nodes = len(node_list)
    node_to_new_idx = {old: new_idx for new_idx, old in enumerate(node_list)}

    # Remap node features (use original's x)
    explain_x = original.x[node_list]

    # Remap edge indices (only if there are edges)
    if explain_edge_index.size(1) > 0:
        new_edge_index = torch.zeros_like(explain_edge_index)
        for i in range(explain_edge_index.size(1)):
            new_u = node_to_new_idx[explain_edge_index[0, i].item()]
            new_v = node_to_new_idx[explain_edge_index[1, i].item()]
            new_edge_index[0, i] = new_u
            new_edge_index[1, i] = new_v
        explain_edge_index = new_edge_index
    # else: already empty

    # Create explanation graph
    # Only include x and edge_index to ensure consistent attributes across all graphs in a batch
    # This prevents KeyError when creating batches with Batch.from_data_list()
    explain_graph = Data(x=explain_x, edge_index=explain_edge_index)

    # Build non-explanation subgraph by calling exclude_explanatory_subgraph
    non_explain_graph = exclude_explanatory_subgraph(original, counterfactual)

    return explain_graph, non_explain_graph


def exclude_explanatory_subgraph(original: Data, counterfactual: Data) -> Data:
    """
    从原图中删除解释子图，返回非解释子图。

    逻辑：
    - 保留所有节点（节点特征来自原图）
    - 保留原图中的边，除非：
      1. 该边在反事实图中不存在，或
      2. 该边的两个端点的特征在反事实图中都发生了变化

    Args:
        original (Data): 原图 (torch_geometric.data.Data)
        counterfactual (Data): 反事实图（节点数与原图相同，一一对应）

    Returns:
        Data: 非解释子图（包含所有节点，但只保留非解释性的边）
    """
    if original.x.size(0) != counterfactual.x.size(0):
        raise ValueError("Original and counterfactual graphs must have the same number of nodes.")

    num_nodes = original.x.size(0)
    device = original.x.device

    # 获取规范化边集合（无向边用 (min, max) 表示）
    def get_canonical_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
        edge_set = set()
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            edge_set.add((min(u, v), max(u, v)))
        return edge_set

    original_edge_set = get_canonical_edge_set(original.edge_index)
    counterfactual_edge_set = get_canonical_edge_set(counterfactual.edge_index)

    # 识别特征发生变化的节点
    changed_nodes: Set[int] = set()
    for i in range(num_nodes):
        if not torch.equal(original.x[i], counterfactual.x[i]):
            changed_nodes.add(i)

    # 收集要保留的边
    keep_edge_indices: List[torch.Tensor] = []
    for i in range(original.edge_index.size(1)):
        u = original.edge_index[0, i].item()
        v = original.edge_index[1, i].item()
        edge_key = (min(u, v), max(u, v))

        # 规则4: 如果边在counterfactual中不存在，不保留（这是解释性的边）
        if edge_key not in counterfactual_edge_set:
            continue

        # 规则6: 如果边的两个端点特征都发生了变化，不保留（这也是解释性的边）
        if u in changed_nodes and v in changed_nodes:
            continue

        # 规则5: 两个图中都存在的边，且不满足规则6，则保留
        keep_edge_indices.append(original.edge_index[:, i:i+1])

    # 构建非解释子图
    if keep_edge_indices:
        keep_edge_index = torch.cat(keep_edge_indices, dim=1)
    else:
        keep_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # 保留所有节点（使用原图的节点特征）
    non_explain_graph = Data(x=original.x.clone(), edge_index=keep_edge_index)

    return non_explain_graph