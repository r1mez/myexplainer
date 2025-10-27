import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from sklearn.cluster import KMeans
from utils.graph_utils import _sanitize_with_valence_correction
def visualize_subgraph(data, edge_mask):
    """
        根据给定的 mask 和 Data 对象，绘制原分子图并把mask部分加粗

        参数：
            data: torch_geometric.data.Data 对象，其中成员的 x 和 edge_attr 为 one-hot 编码
            edge_mask: data的子图mask，以torch.tensor存储
        """
    atom_map = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K',
                12: 'Li', 13: 'Ca'}
    bond_map = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

    # 提取原子类型
    atom_indices = torch.argmax(data.x, dim=1).tolist()
    atoms = [atom_map[idx] for idx in atom_indices]

    # 创建可编辑分子
    editable_mol = Chem.RWMol()

    # 添加原子
    for atom_symbol in atoms:
        atom = Chem.Atom(atom_symbol)
        editable_mol.AddAtom(atom)

    # 提取边和键类型
    edge_index = data.edge_index.t().tolist()  # [[source, target], ...]
    bond_indices = torch.argmax(data.edge_attr, dim=1).tolist()
    bonds = [bond_map[idx] for idx in bond_indices]

    # 使用集合来跟踪已添加的键（避免重复）
    added_bonds = set()
    highlighted_bonds = []

    for i, (source, target) in enumerate(edge_index):
        # 标准化键顺序，确保 source < target 以避免重复
        bond_key = tuple(sorted([source, target]))
        if bond_key not in added_bonds:
            bond_type = bonds[i]
            editable_mol.AddBond(min(source, target), max(source, target), bond_type)
            added_bonds.add(bond_key)

            # 获取当前键的索引（添加后的键数量 - 1）
            bond_idx = editable_mol.GetMol().GetNumBonds() - 1

            # 检查 mask：由于双向边，检查当前 i 和反向边的 mask（如果存在）
            # 找到反向边的索引
            reverse_idx = None
            for j, (s, t) in enumerate(edge_index):
                if s == target and t == source:
                    reverse_idx = j
                    break

            # 如果当前或反向边的 mask 为 True，则高亮
            if edge_mask[i] or (reverse_idx is not None and edge_mask[reverse_idx]):
                highlighted_bonds.append(bond_idx)

    # 获取最终分子
    mol = editable_mol.GetMol()

    # 清理分子（推荐）
    # Chem.SanitizeMol(mol)
    _sanitize_with_valence_correction(mol)
    # mol = Chem.RemoveHs(mol,sanitize=False)

    # 绘制分子，高亮 mask 部分的键（加粗，使用红色高亮）
    img = MolToImage(mol, size=(600, 600), highlightBonds=highlighted_bonds, highlightColor=(0.8, 0.8, 0.8))
    img.save('molecule.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return img