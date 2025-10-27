from networkx.classes import subgraph
from rdkit import Chem

# MAX_VALENCE = {"C": 4,"O": 2,"Cl": 7,"H": 1,"N": 5,"F": 1,"Br": 7,"S": 6,"P": 5,"I": 7}     #Mutag
MAX_VALENCE = {"H": 1, "B": 3, "C": 4, "N": 5, "O": 2, "F": 1, "Na": 1, "P": 5, "S": 6, "Cl":7, "Ca": 2, "Br": 7, "I": 7}  # BBBP

def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol, canonical=True):
    return Chem.MolToSmiles(mol, canonical=canonical)

def mol2sma(mol):
    return Chem.MolToSmarts(mol)
def get_submol(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        if atom_symbol == 'Si':
            atom_symbol = '[Si]'
        elif atom_symbol == 'Na':
            atom_symbol = '[Na+]'
        return smi2mol(atom_symbol, kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # turn to smiles order
    smi = mol2smi(submol)
    submol = smi2mol(smi, kekulize, sanitize=False)
    # # special with N+ and N-
    # for atom in submol.GetAtoms():
    #     if atom.GetSymbol() != 'N':
    #         continue
    #     if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
    #         atom.SetNumRadicalElectrons(0)
    #         atom.SetNumExplicitHs(2)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new


def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())

def is_substructure(smarts, smiles):
    submol = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or submol is None:
        return False
    return mol.HasSubstructMatch(submol)

#
# if __name__ == '__main__':
#     with open('../../../data/mutag/smiles/mutag_smiles_0.txt','r',encoding='utf-8') as f:
#         mol_list = f.read().splitlines()
#     print(mol_list)
#     cnt=0
#     for mol in mol_list:
#         if is_substructure('c1ccc2cc3ccccc3cc2c1', mol):
#             print(mol+f'   {cnt+1}')
#             cnt+=1
#     print(cnt)
