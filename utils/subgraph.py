from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

from tqdm import tqdm


class Fragment:
    def __init__(self, smiles: str, atom_indices: List[int]):
        self.smiles = smiles
        self.atom_indices = sorted(atom_indices)  # Keep sorted for consistency


def build_submol(mol: Chem.Mol, atom_indices: List[int]) -> Chem.Mol:
    """Build the induced subgraph for given atom indices, avoiding strict sanitization."""
    if not atom_indices:
        return None
    rw = Chem.RWMol(Chem.MolFromSmiles(''))
    atom_map = {}
    for idx in atom_indices:
        atom = Chem.Atom(mol.GetAtomWithIdx(idx))
        new_idx = rw.AddAtom(atom)
        atom_map[idx] = new_idx
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_map and a2 in atom_map:
            bt = bond.GetBondType()
            rw.AddBond(atom_map[a1], atom_map[a2], bt)
    # Avoid full sanitization to prevent kekulization errors; use minimal sanitization
    try:
        Chem.SanitizeMol(rw, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        return rw.GetMol()  # Return unsanitized mol if kekulization fails
    return rw.GetMol()


def validate_smiles(smiles: str) -> Chem.Mol:
    """Validate and sanitize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def principal_subgraph_extraction(smiles_list: List[str], N: int) -> List[str]:
    """
    Implementation of Algorithm 1: Principal Subgraph Extraction with error handling.

    Args:
        smiles_list: List of SMILES strings for the dataset D.
        N: Desired number of principal subgraphs.

    Returns:
        V: List of SMILES strings for the extracted principal subgraphs.
    """
    # Pre-process and validate SMILES
    D: List[Tuple[Chem.Mol, List[Fragment]]] = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        # if mol is None:
        #     print('无效的SMILES:', s)
        #     continue
        fragments = [Fragment(mol.GetAtomWithIdx(i).GetSymbol(), [i]) for i in range(mol.GetNumAtoms())]
        D.append((mol, fragments))

    # Initialize V with unique atoms
    V_set = set()
    for _, fragments in D:
        for frag in fragments:
            V_set.add(frag.smiles)
    V = list(V_set)
    n_prime = max(N, len(V))
    pbar = tqdm(total=n_prime, desc="Extracting subgraphs")
    while len(V) < n_prime:
        counter = Counter()
        for mol, fragments in D:
            # Build atom to fragment map
            frag_map = {}
            for frag in fragments:
                for idx in frag.atom_indices:
                    frag_map[idx] = frag

            # Find unique neighboring fragment pairs
            neighbor_pairs = set()
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                f1 = frag_map[a1]
                f2 = frag_map[a2]
                if f1 is not f2:
                    pair_key = frozenset({id(f1), id(f2)})
                    neighbor_pairs.add(pair_key)

            # For each unique pair, merge and count
            for pair_key in neighbor_pairs:
                pair_frags = [f for f in fragments if id(f) in pair_key]
                if len(pair_frags) != 2:
                    continue
                f1, f2 = pair_frags
                all_indices = f1.atom_indices + f2.atom_indices
                submol = build_submol(mol, all_indices)
                if submol:
                    merged_smiles = Chem.MolToSmiles(submol, canonical=True)
                    counter[merged_smiles] += 1

        if not counter:
            break

        # Select most frequent
        most_freq_smiles = counter.most_common(1)[0][0]
        V.append(most_freq_smiles)

        # Update D
        new_D = []
        new_F_mol = validate_smiles(most_freq_smiles)
        if new_F_mol is None:
            continue
        for mol, fragments in D:
            matches = mol.GetSubstructMatches(new_F_mol)
            if not matches:
                new_D.append((mol, [deepcopy(f) for f in fragments]))
                continue

            sorted_matches = sorted(matches, key=lambda m: min(m))
            covered = set()
            covered_frags = set()
            new_fragments = []

            for match in sorted_matches:
                match_set = set(match)
                if not covered.isdisjoint(match_set):
                    continue
                involved_frags = set()
                union_indices = set()
                for frag in fragments:
                    frag_set = set(frag.atom_indices)
                    if frag_set & match_set:
                        involved_frags.add(frag)
                        union_indices.update(frag_set)
                if union_indices == match_set:
                    new_frag = Fragment(most_freq_smiles, list(match_set))
                    new_fragments.append(new_frag)
                    covered.update(match_set)
                    covered_frags.update(involved_frags)
                # Else skip, as it doesn't cover whole fragments

            for frag in fragments:
                if frag not in covered_frags:
                    new_fragments.append(deepcopy(frag))

            new_D.append((mol, new_fragments))

        D = new_D
        pbar.update(1)

    return V