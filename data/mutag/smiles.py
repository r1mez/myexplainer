import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import re


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


def data_to_smiles(data: Data) -> str:
    """
    Convert a torch_geometric.data.Data object representing a molecule to a SMILES string.

    Args:
        data (Data): A torch_geometric Data object with:
            - x: Node features (one-hot encoded, shape [num_nodes, 14], for atoms)
            - edge_index: Edge indices representing atom connections (shape [2, num_edges])
            - edge_attr: Edge attributes (integer labels or one-hot for bond types, e.g., 0=single, 1=double, 2=triple)

    Returns:
        str: SMILES string representing the molecule

    Raises:
        ValueError: If the molecule is invalid or cannot be converted to SMILES
    """
    #如果data有idx属性，则将其赋值给idx，否则idx为None
    idx = data.idx if hasattr(data, 'idx') else None
    if idx == 2262:
        return 'CN'
    if idx == 1431:
        return 'CCCN=NO'
    if idx == 2687:
        return 'CCN=NO'
    if idx == 150:
        return 'CCCCN(CC(O)C1=CC(=[N+]=[N-])C(=O)C=C1)N=O'
    if idx == 880:
        return '[N-]=[N+]=C1C=CC(=O)C=C1'
    if idx == 1260:
        return 'CCOC(=O)CNC(=O)C=[N+]=[N-]'
    if idx == 1998:
        return '[N-]=[N+]=CC(=O)NCC(N)=O'
    if idx == 2115:
        return '[N-]=[N+]=CC(=O)OCC(N)C(=O)O'
    if idx == 3747:
        return '[N-]=[N+]=CC(=O)NCC(=O)NN'
    if idx == 4034:
        return '[N-]=[N+]=C1C=NC(=O)NC1=O'
    # zwitter ions
    if idx == 404:
        return 'C[N+](C)(C)CC(=O)[O-]'
    if idx == 2701:
        return 'C[n+]1cccc(C(=O)[O-])c1'
    if idx == 3051:
        return 'CCN(CC)c1ccc2c(-c3ccc(S(=O)(=O)[O-])cc3S(=O)(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1'
    if idx == 3554:
        return 'CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2c(O)c(S(=O)(=O)[O-])cc3cc(S(=O)(=O)O)ccc23)cc1'
    if idx == 497:
        return 'CCN(Cc1cccc(S(=O)(=O)[O-])c1)c1ccc(C(=C2C=CC(=[N+](CC)Cc3cccc(S(=O)(=O)O)c3)C=C2)c2ccc(N(C)C)cc2)cc1'
    if idx == 3448:
        return 'C[N+](C)(C)NCCC(=O)[O-]'
    # missing double bond
    if idx == 2564:
        return 'Cn1cnc2c1nc[n+]([O-])C=2N'
    # remove ammonia
    if idx == 1674:
        return 'NC(CCC(=O)O)C(=O)O'
    # two pos nitrogen neighbors also make zwitter ion
    if idx == 1153:
        return 'C=C(C)C([O-])=N[N+](C)(C)CC(C)O'
    if idx == 3448:
        return 'C[N+](C)(C)NCCC(=O)[O-]'
    # large fragments
    if idx == 1233:
        return 'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2nc(C(=O)NCCC[S+](C)C)cs2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1'
    # ionic
    if idx == 490:
        return 'Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1'
    if idx == 638:
        return 'Cn1c(-c2ccccc2)c(N=Nc2scc[n+]2C)c2ccccc21'
    if idx == 1125:
        return 'CCN(CC)c1ccc2nc3ccc(N(CC)CC)cc3[o+]c2c1'
    if idx == 1151:
        return 'CC(C)[N+](C)(CCOC(=O)C1c2ccccc2Oc2ccccc21)C(C)C'
    if idx == 1369:
        return 'CC[n+]1c(-c2ccccc2)c2cc(N)ccc2c2ccc(N)cc21'
    if idx == 1379:
        return 'ClC=CC[N+]12CN3CN(CN(C3)C1)C2'
    if idx == 1484:
        return 'Cc1cc2nc3cc(C)c(N)cc3[n+](-c3ccccc3)c2cc1N'
    if idx == 1508:
        return 'O=[N+]([O-])c1ccc(-n2nc(-c3ccccc3)n[n+]2-c2ccc(I)cc2)cc1'
    if idx == 1743:
        return '[O-][N+]([O-])=C1CCCC1'
    if idx == 1845:
        return 'C[n+]1ccc(-c2cc[n+](C)cc2)cc1'
    if idx == 2972:
        return 'C[n+]1cccc2c1C=CC1OC21'
    if idx == 3108:
        return 'N#[N+]c1ccc([N+](=O)[O-])cc1'
    if idx == 3376:
        return 'CC(C)=[N+]([O-])[O-]'
    if idx == 3387:
        return 'C[n+]1c2cc(N)ccc2cc2ccc(N)cc21'
    if idx == 3408:
        return 'C[N+](C)(C)CCNCCc1ccc(N=Nc2ccc([N+](=O)[O-])cc2Cl)cc1'
    if idx == 3484:
        return 'CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(N(CC)CC)cc3[o+]c2c1'
    if idx == 3566:
        return 'CCN(CC)c1ccc(C(=C2C=CC(=[N+](CC)CC)C=C2)c2ccc(N(CC)CC)cc2)cc1'
    if idx == 3908:
        return 'C[n+]1ccc(-c2ccccc2)cc1'
    if idx == 3946:
        return 'CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1'
    # large counter ions
    if idx == 58:
        return 'O=C(O)COc1ccc(Cl)cc1Cl'
    if idx == 138:
        return 'CN1CCN(C2=Nc3ccccc3Oc3ccc(Cl)cc32)CC1'
    if idx == 144:
        return 'CC(C)(C)NCC(O)COc1nsnc1N1CCOCC1'
    if idx == 148:
        return 'CNC(=O)Oc1ccc2c(c1)C1(C)CCN(C)C1N2C'
    if idx == 182:
        return 'O=c1cn[nH]c(=O)[nH]1'
    if idx == 219:  # also ionic
        return 'CCCCCCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1'
    if idx == 386:  # also ionic
        return 'NC(=O)c1cc[n+](COC[n+]2ccccc2C=NO)cc1'
    if idx == 464:
        return 'NNc1nnc(NN)c2ccccc12'
    if idx == 478:
        return 'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12'
    if idx == 560:
        return 'CS(=O)(=O)OCCCNCCCOS(C)(=O)=O'
    if idx == 567:
        return 'CN1Cc2c(N)cccc2C(c2ccccc2)C1'
    if idx == 640:
        return 'COc1ccc(N)cc1N'
    if idx == 667:  # also ionic
        return 'CCOCC(O)COc1ccc(NC(=O)CC[S+](C)C)cc1'
    if idx == 872:  # also ionic
        return 'C[n+]1c2ccccc2nc2ccccc21'
    if idx == 902:
        return 'OCc1c2ccccc2cc2c1ccc1ccccc12'
    if idx == 1019:  # also ionic
        return 'Cc1ccc([N+]#N)cc1'
    if idx == 1052:
        return 'CNC(C)C(O)c1ccccc1'
    if idx == 1087:  # also ionic
        return 'CN(C)C(=O)Oc1ccc[n+](C)c1'
    if idx == 1233:  # also ionic
        return  'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2nc(C(=O)NCCC[S+](C)C)cs2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1'
    if idx == 1307:  # also ionic
        return 'CCCCCCCCCCSCn1cc[n+](C)c1'
    if idx == 1375:
        return 'COc1cc(NC(C)CCCN)c2ncccc2c1'
    if idx == 1525:
        return 'CN1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc(c34)CC21'
    if idx == 1587:  # also ionic
        return 'CC[N+](CC)=c1ccc2nc3c(cc(N)c4ccccc43)oc-2c1'
    if idx == 1626:
        return 'C1CCC(C(CC2CCCCN2)C2CCCCC2)CC1'
    if idx == 1654:
        return 'CN(C)CCC(c1ccccc1)c1ccccn1'
    if idx == 1691:
        return 'COc1ccc(CN(CCN(C)C)c2ccccn2)cc1'
    if idx == 1777:
        return 'O=c1cn[nH]c(=O)[nH]1'
    if idx == 1949:
        return 'CC(=C(OCCOc1ccc(Cl)cc1)c1ccc(Cl)cc1Cl)n1ccnc1'
    if idx == 2095:  # also ionic
        return 'Oc1cc(O)c2cc(O)c(-c3cc(O)c(O)c(O)c3)[o+]c2c1'
    if idx == 2330:
        return 'CCOc1ccc2nc3cc(N)ccc3c(N)c2c1'
    if idx == 2360:
        return 'CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O'
    if idx == 2371:
        return 'Oc1cccc2cccnc12'
    if idx == 2485:  # also ionic
        return 'C[N+](C)(C)Cc1ccccc1'
    if idx == 2513:  # also ionic
        return 'CCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1'
    if idx == 2639:
        return 'CCN(CC)CCNc1ccc(CO)c2sc3ccccc3c(=O)c12'
    if idx == 2694:
        return 'Cc1cc(N)ccc1N'
    if idx == 2762:
        return 'O=c1cn[nH]c(=O)[nH]1'
    if idx == 2777:  # also ionic
        return 'CC[n+]1c2ccccc2nc2ccccc21'
    if idx == 2797:
        return 'CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1'
    if idx == 2868:
        return 'CCCn1cc2c3c(cccc31)C1C=C(C)CN(C)C1C2'
    if idx == 2956:  # also ionic
        return 'C[n+]1c2ccccc2c(N)c2ccccc21'
    if idx == 3020:
        return 'Nc1ccc2cc3ccc(N)cc3nc2c1'
    if idx == 3047:
        return 'N=c1ccc2nc3c(cc(N)c4ccccc43)oc-2c1'
    if idx == 3071:  # also ionic
        return 'C[n+]1c2ccccc2cc2ccccc21'
    if idx == 3081:
        return 'Clc1ccc(COC(Cn2ccnc2)c2ccc(Cl)cc2Cl)cc1'
    if idx == 3135:  # also ionic
        return 'CCN(CC)c1ccc(C(=C2C=CC(=[N+](CC)CC)C=C2)c2ccccc2)cc1'
    if idx == 3217:
        return 'CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C)C3C(O)(C(=O)OC)C(OC(C)=O)C5(CC)C=CCN6CCC43C65)C2)C1'
    if idx == 3279:
        return 'O=P(O)(NCCCO)N(CCCl)CCCl'
    if idx == 3294:
        return 'CCC(=O)N(c1ccc(Cl)c(Cl)c1)C1CCCC1N(C)C'
    if idx == 3340:
        return 'CN(C)CCOC(C)(c1ccccc1)c1ccccn1'
    if idx == 3362:  # also ionic
        return 'C[n+]1c2ccccc2cc2c(N)cccc21'
    if idx == 3371:
        return 'CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C=O)C3C(O)(C(=O)OC)C(OC(C)=O)C5(CC)C=CCN6CCC43C65)C2)C1'
    if idx == 3424:  # also ionic
        return 'C[n+]1c2ccccc2cc2cc(N)ccc21'
    if idx == 3494:  # also ionic
        return 'N#[N+]c1cccc2c1C(=O)c1ccccc1C2=O'
    if idx == 3503:  # also ionic
        return 'C[N+]1=CC=C(c2ccccc2)CC1'
    if idx == 3594:
        return 'N=C(N)c1ccc(OCCCCCOc2ccc(C(=N)N)cc2)cc1'
    if idx == 3719:  # also ionic
        return 'C[n+]1c2ccccc2cc2ccc(N)cc21'
    if idx == 3860:
        return 'CNc1ccc(O)cc1'
    if idx == 3919:
        return 'COc1ccc2c3c1OC1C(O)C=CC4C(C2)N(C)CCC341'
    if idx == 4054:
        return 'O=C(O)COc1ccc(Cl)cc1Cl'
    if idx == 4239:  # also ionic
        return 'C[N+]1(C)CCOCC1'
    if idx == 4317:
        return 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)O'
    # change compounds with [SH] to [S+]
    if idx == 278:
        return 'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2ncc(C(=O)NCCC[S+](C)C)s2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1'
    if idx == 2511:
        return 'Cc1cc2c(cc1N)=[S+]c1cc(N(C)C)ccc1N=2'
    
    # Define atom mapping based on README
    atom_map = {
        0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F',
        6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K',
        12: 'Li', 13: 'Ca'
    }

    # Define bond mapping based on README
    bond_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE
    }

    # Create an editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for node_idx in range(data.x.size(0)):
        node_feature = data.x[node_idx]
        if node_feature.dim() == 1 and node_feature.size(0) == 14:
            # One-hot encoded node features
            atom_idx = torch.argmax(node_feature).item()
        else:
            raise ValueError(f"Unexpected node feature shape: {node_feature.shape}")

        atom_type = atom_map[atom_idx]
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)

    # Process edges to remove duplicates (handle undirected graph)
    seen_edges = set()
    for edge_idx in range(data.edge_index.size(1)):
        src, dst = data.edge_index[:, edge_idx].tolist()
        edge = tuple(sorted([src, dst]))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)

        # Handle edge attributes
        edge_feature = data.edge_attr[edge_idx]
        if edge_feature.dim() == 1 and edge_feature.size(0) > 1:
            # One-hot encoded bonds
            bond_idx = torch.argmax(edge_feature).item()
        else:
            # Single integer label
            bond_idx = edge_feature.item() if edge_feature.dim() == 0 else edge_feature[0].item()

        bond_type = bond_map[bond_idx]
        if src >= 0 and dst >= 0:
            mol.AddBond(src, dst, bond_type)
        else:
            print("---------")
            raise ValueError(f"Invalid bond indices: src={src}, dst={dst}")

    # Convert to RDKit molecule and sanitize
    try:
        mol = mol.GetMol()
        _sanitize_with_valence_correction(mol)  # Use custom sanitization with valence correction
        mol = Chem.RemoveHs(mol, sanitize=False)
        smiles = Chem.MolToSmiles(mol)

        if '.' in smiles:  # 判断是否为混合物
            frags = smiles.split('.')  # 分割混合物
            long_parts = [frag for frag in frags if len(frag) > 7 ]  # 找出长度大于7的部分

            if len(long_parts) == 1:  # 只有一个长度大于7的部分
                return long_parts[0]
            else:
                return ""
        else:  # 不是混合物
            return smiles
    except Exception as e:
        raise ValueError(f"Failed to convert molecule to SMILES: {str(e)}")
    
