import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdkit.Chem import rdchem

from rdkit import RDLogger
# def _correct_valence(mol: Chem.RWMol, c_atom: Chem.Atom) -> None:
#     """
#     Correct valence and charge for a specific atom in the molecule.
#     Based on the reference code's logic for handling nitrogen and oxygen atoms.
#
#     Args:
#         mol: RDKit molecule (RWMol)
#         c_atom: The atom to correct (Chem.Atom)
#     """
#     neighbors = c_atom.GetNeighbors()
#     neighbor_types = [n.GetSymbol() for n in neighbors]
#
#     if c_atom.GetSymbol() == "N":
#         # Nitro group (2 or 3 oxygen neighbors)
#         if neighbor_types.count("O") > 1:
#             for n in neighbors:
#                 bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
#                 if (n.GetSymbol() == "O" and
#                         bond.GetBondTypeAsDouble() == 1.0 and
#                         len(n.GetNeighbors()) == 1):
#                     c_atom.SetFormalCharge(1)
#                     n.SetFormalCharge(-1)
#
#         # RN(O)=NR or R2[N+]=O
#         elif neighbor_types.count("O") == 1:
#             for n in neighbors:
#                 if n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
#                         c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1:
#                     c_atom.SetFormalCharge(1)
#                     n.SetFormalCharge(-1)
#                 elif n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
#                         c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2:
#                     c_atom.SetFormalCharge(1)
#
#         # Diazo group
#         elif neighbor_types.count("N") == 1:
#             bonds = [b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]
#             if 3.0 in bonds:  # C-[N+]#[N]
#                 c_atom.SetFormalCharge(1)
#             elif len(bonds) <= 3:  # Ammonium cation
#                 c_atom.SetFormalCharge(1)
#             else:  # C=[N+]=[N-]
#                 for n in neighbors:
#                     if n.GetSymbol() == "N":
#                         c_atom.SetFormalCharge(1)
#                         n.SetFormalCharge(-1)
#
#         # Azides
#         elif neighbor_types.count("N") > 1:
#             for n in neighbors:
#                 if n.GetSymbol() == "N" and len([nn.GetSymbol() for nn in n.GetNeighbors()]) == 1:
#                     c_atom.SetFormalCharge(1)
#                     n.SetFormalCharge(-1)
#                 else:
#                     c_atom.SetFormalCharge(1)
#
#         # Ammonium cations
#         elif neighbor_types.count("C") >= 3:
#             c_atom.SetFormalCharge(1)
#
#         else:
#             raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")
#
#     elif c_atom.GetSymbol() == "O":
#         # Oxygen with single and double bonds
#         if set([b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]) == {1.0, 2.0}:
#             c_atom.SetFormalCharge(1)
#         else:
#             raise ValueError(f"Unexpected group at atom {c_atom.GetIdx()}")
#
#     else:
#         raise ValueError(f"Unexpected atom {c_atom.GetSymbol()} with idx {c_atom.GetIdx()}")
#
#
# def _sanitize_with_valence_correction(mol: Chem.RWMol) -> None:
#     """
#     Sanitize molecule and correct valence/charge if needed.
#
#     Args:
#         mol: RDKit molecule (RWMol)
#
#     Raises:
#         ValueError: If sanitization or valence correction fails
#     """
#     try:
#         Chem.SanitizeMol(mol)
#     except Chem.rdchem.AtomValenceException as e:
#         match = re.search(r"atom # (\d+)", str(e))
#         if match:
#             atom_idx = int(match.group(1))
#             _correct_valence(mol, mol.GetAtomWithIdx(atom_idx))
#             _sanitize_with_valence_correction(mol)  # Recursive call after correction
#         else:
#             raise ValueError(f"No atom number in exception: {str(e)}")


def nci12smiles(data: Data) -> str:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        # Some dataset does not have
        atom = rdchem.Atom(torch.argmax(data.x[i]).item() + 1)
        mol.AddAtom(atom)
    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()
    deleted = []

    # print(f"Data: {data}")
    # print(f"Data attribute: {data.keys}")

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue
        mol.AddBond(src, dst)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    # if kekulize:
    #     Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = sanitize(mol)
    if mol is None:
        # import ipdb; ipdb.set_trace()
        return None
    # return get_smiles(mol)
    # Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_smiles(mol):
    RDLogger.DisableLog('rdApp.*')
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def get_mol(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True) # Add clearAromaticFlags to avoid error
    if mol is None:
        return None
    return mol