"""Chemistry utilities: atom/bond maps, RDKit molecular conversions."""

import re

import torch
from rdkit import Chem
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Atom maps
# ---------------------------------------------------------------------------
MUTAG_atom_map = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
MUTAG188_atom_map = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
BBBP_atom_map = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 15: 'P', 16: 'S', 17: 'Cl', 20: 'Ca', 35: 'Br', 53: 'I'}
atom_map = {"mutag": MUTAG_atom_map, "mutag188": MUTAG188_atom_map, "bbbp": BBBP_atom_map}

# ---------------------------------------------------------------------------
# Bond maps
# ---------------------------------------------------------------------------
MUTAG_bond_map = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}
MUTAG188_bond_map = {0: Chem.BondType.AROMATIC, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
BBBP_bond_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
bond_map = {"mutag": MUTAG_bond_map, "mutag188": MUTAG188_bond_map, "bbbp": BBBP_bond_map}

# ---------------------------------------------------------------------------
# Index maps
# ---------------------------------------------------------------------------
MUTAG_idx_map = {'C': 0, 'O': 1, 'Cl': 2, 'H': 3, 'N': 4, 'F': 5, 'Br': 6, 'S': 7, 'P': 8, 'I': 9, 'Na': 10, 'K': 11,
                 'Li': 12, 'Ca': 13}
MUTAG188_idx_map = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'I': 4, 'Cl': 5, 'Br': 6}
atom_idx_map = {"mutag": MUTAG_idx_map, "mutag188": MUTAG188_idx_map}


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

    if dataset_name not in atom_map or dataset_name not in atom_idx_map:
        raise ValueError(f"SMARTS conversion is not configured for dataset {dataset_name}")

    # Get atom features (one-hot encoding based on dataset atom map)
    num_atoms = mol.GetNumAtoms()
    num_atom_types = len(atom_map[dataset_name])
    atom_features = []
    dataset_atom_idx_map = atom_idx_map[dataset_name]

    for atom in mol.GetAtoms():
        # Get atom symbol
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_map[dataset_name].values():
            raise ValueError(f"Atom type {atom_symbol} not found in {dataset_name}_atom_map")

        # Create one-hot encoding
        one_hot = [0] * num_atom_types
        one_hot[dataset_atom_idx_map[atom_symbol]] = 1
        atom_features.append(one_hot)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices and edge attributes
    edge_index = []
    edge_attr = []
    num_edge_types = len(bond_map[dataset_name])
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
            if dataset_name == "mutag188":
                mapped_bond_type = 0
            else:
                # Aromatic bond: alternate between single (0) and double (1)
                if bond_idx not in bond_alternation:
                    # Alternate based on bond index to ensure consistency
                    bond_alternation[bond_idx] = 0 if len(bond_alternation) % 2 == 0 else 1
                mapped_bond_type = bond_alternation[bond_idx]
        elif bond_type in [1.0, 2.0, 3.0]:
            if dataset_name == "mutag188":
                # Classic MUTAG keeps aromatic as 0, then single/double/triple as 1/2/3.
                mapped_bond_type = int(bond_type)
            else:
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
        if dataset_name in {"mutag", "mutag188"}:
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
        if dataset_name in {"mutag", "mutag188"}:
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
