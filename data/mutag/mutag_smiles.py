# %%
import torch
from torch_geometric.datasets import TUDataset
from rdkit.Chem import AllChem as Chem
import networkx as nx
from matplotlib import pyplot as plt
import re
import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
from rdkit.Chem import AllChem as Chem
from datasets.mutag_dataset import Mutagenicity


class MolGenMutagenicity():
    def __init__(self):
        self.atom_types = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S",
                           8: "P", 9: "I", 10: "Na", 11: "K", 12: "Li", 13: "Ca", }
        self.bond_types = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

    def _get_atoms(self, mol, x):
        atoms = []
        for atom_vec in x:
            internal_atom_type = torch.argmax(atom_vec).item()
            atom = Chem.Atom(self.atom_types[internal_atom_type])
            atoms.append(atom)
            mol.AddAtom(atom)
        assert mol.GetNumAtoms() == x.shape[
            0], f"Atom count missmatch: {x.shape[0]} (Torch), {mol.GetNumAtoms()} (RDKit)"

    def _get_bonds(self, mol, edge_index, edge_attr):
        assert edge_index.shape[1] == edge_attr.shape[0], f"Different number of bonds in edge_index and edge_attr."
        track_bonds = []
        for edge, edge_type in zip(edge_index.t(), edge_attr):
            if set(edge.tolist()) in track_bonds:
                continue
            else:
                edge = edge.tolist()
                track_bonds.append(set(edge))
                bond_type = self.bond_types[torch.argmax(edge_type).item()]
                mol.AddBond(*edge, bond_type)
        assert mol.GetNumBonds() == edge_index.shape[
            1] / 2, f"Bond count missmatch: {edge_index.shape[1] / 2} (Torch), {mol.GetNumBonds()} (RDKit)"

    def _sanitize_with_valence_correnction(self, mol):
        try:
            Chem.SanitizeMol(mol)
        except Chem.rdchem.AtomValenceException as e:
            match = re.search(r"atom # (\d+)", e.args[0])
            if match:
                atom_idx = int(match.group(1))
            else:
                assert False, f"no atom number for exception: {e}"
            # convert to correct group (mostly nitrogen containing groups)
            self._correct_valence(mol, mol.GetAtomWithIdx(atom_idx))
            self._sanitize_with_valence_correnction(mol)

    def _correct_valence(self, mol, c_atom):
        neighbors = c_atom.GetNeighbors()
        neighbor_types = [n.GetSymbol() for n in neighbors]
        if c_atom.GetSymbol() == "N":
            # 2 neighbors == O -> *NITRO GROUP*
            # 3 neighbours == O -> *NITRATE ESTER*
            if neighbor_types.count("O") > 1:
                for n in neighbors:
                    bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), n.GetIdx())
                    if (n.GetSymbol() == "O"
                            and bond.GetBondTypeAsDouble() == 1.0
                            and len(n.GetNeighbors()) == 1):
                        c_atom.SetFormalCharge(1)
                        n.SetFormalCharge(-1)

            # weird RN(O)=NR group or R2[N+]=O
            elif neighbor_types.count("O") == 1:
                for n in neighbors:
                    if n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                            c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1:
                        c_atom.SetFormalCharge(1)
                        n.SetFormalCharge(-1)
                    elif n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(
                            c_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2:
                        c_atom.SetFormalCharge(1)

            # *DIAZO GROUP*
            elif neighbor_types.count("N") == 1:
                bonds = [b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]
                if 3.0 in bonds:  # C-[N+]#[N] # make cation
                    c_atom.SetFormalCharge(1)
                elif len(bonds) <= 3:  # ammonium cation
                    c_atom.SetFormalCharge(1)
                else:  # C=[N+]=[N-]
                    for n in neighbors:
                        if n.GetSymbol() == "N":
                            c_atom.SetFormalCharge(1)
                            n.SetFormalCharge(-1)

            # *AZIDES*
            elif neighbor_types.count("N") > 1:
                for n in neighbors:
                    if n.GetSymbol() == "N" and len([nn.GetSymbol() for nn in n.GetNeighbors()]) == 1:
                        c_atom.SetFormalCharge(1)
                        n.SetFormalCharge(-1)
                    else:
                        c_atom.SetFormalCharge(1)

            # *AMMONIUM CATIONS*
            elif neighbor_types.count("C") >= 3:
                c_atom.SetFormalCharge(1)

            else:
                assert False, f"Unexpected group at atom {c_atom.GetIdx()}"

        elif c_atom.GetSymbol() == "O":
            # make cation
            if set([b.GetBondTypeAsDouble() for b in c_atom.GetBonds()]) == {1.0, 2.0}:
                c_atom.SetFormalCharge(1)
            else:
                assert False, f"Unexpected group at atom {c_atom.GetIdx()}"
        else:
            assert False, f"Unexpected atom {c_atom.GetSymbol()} with idx {c_atom.GetIdx()}"

    def get_mol(self, pyg_data_object):
        mol = Chem.RWMol()
        x = pyg_data_object.x  # one-hot encoded
        edge_index = pyg_data_object.edge_index
        edge_attr = pyg_data_object.edge_attr  # one-hot encoded
        y = pyg_data_object.y.item()  # 0 = mutagen; 1 = nonmutagen

        self._get_atoms(mol, x)
        self._get_bonds(mol, edge_index, edge_attr)
        self._sanitize_with_valence_correnction(mol)
        mol.SetProp("Mutagenicity", "nonmutagen" if y else "mutagen")
        return Chem.Mol(mol)  # convert to regular mol object

    def get_smiles(self, pyg_data_object, with_explicit_Hs=False):
        if with_explicit_Hs:
            return Chem.MolToSmiles(self.get_mol(pyg_data_object))
        else:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(
                    Chem.MolToSmiles(
                        self.get_mol(pyg_data_object))))

    def get_unsanitized_mol(self, pyg_data_object):
        mol = Chem.RWMol()
        x = pyg_data_object.x  # one-hot encoded
        edge_index = pyg_data_object.edge_index
        edge_attr = pyg_data_object.edge_attr  # one-hot encoded
        y = pyg_data_object.y.item()  # 0 = mutagen; 1 = nonmutagen
        self._get_atoms(mol, x)
        self._get_bonds(mol, edge_index, edge_attr)
        mol.SetProp("Mutagenicity", "nonmutagen" if y else "mutagen")
        for atom in mol.GetAtoms():
            atom.SetProp('atomNote', str(atom.GetIdx()))
        return Chem.Mol(mol)  # convert to regular mol object

    def get_networkx(self, pyg_data_object):
        x = pyg_data_object.x  # one-hot encoded
        edge_index = pyg_data_object.edge_index
        edge_attr = pyg_data_object.edge_attr  # one-hot encoded
        graph = nx.Graph()
        atoms = []
        for i, atom_vec in enumerate(x):
            internal_atom_type = torch.argmax(atom_vec).item()
            # atom = {i: self.atom_types[internal_atom_type]}
            atom = self.atom_types[internal_atom_type] + f"_{i}"
            atoms.append(atom)
            graph.add_node(atom)
        track_bonds = []
        for edge, edge_type in zip(edge_index.t(), edge_attr):
            if set(edge.tolist()) in track_bonds:
                continue
            else:
                edge = edge.tolist()
                track_bonds.append(set(edge))
                graph.add_edge(atoms[edge[0]], atoms[edge[1]], bond=torch.argmax(edge_type).item() + 1)
        return graph

    def draw_networkx(self, pyg_data_object):
        G = self.get_networkx(pyg_data_object)
        pos = nx.spring_layout(G, k=1 / 2)
        color_map = []
        for node in G.nodes():
            atom = str(node).split("_")[0]
            if atom == "C":
                color_map.append('darkgrey')
            elif atom == "N":
                color_map.append('blue')
            elif atom == "O":
                color_map.append('red')
            elif atom == "H":
                color_map.append('whitesmoke')
            else:
                color_map.append('darkgreen')
        nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=700)
        nx.draw_networkx_edges(G, pos)
        bonds = nx.get_edge_attributes(G, 'bond')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=bonds)
        plt.show()


class ManualFixer():
    def __init__(self, mutagenicity_dataset):
        self.data = mutagenicity_dataset
        self.gen = MolGenMutagenicity()
        self.faulty_mol_idxs = self.find_faulty_mols()
        self.smiles_data_raw = None
        self.smiles_single_mols_no_ions = None

    def find_faulty_mols(self):
        mols = [self.gen.get_mol(d) for d in self.data]
        faulty_mol_idxs = []
        for i, m in enumerate(mols):
            if Chem.GetFormalCharge(m) != 0 or len(Chem.GetMolFrags(m)) > 1:
                faulty_mol_idxs.append(i)
        return faulty_mol_idxs

    def get_smiles_data_raw(self):
        mols = [self.gen.get_mol(d) for d in self.data]
        smiles_csv = {"smiles": [], "mutagenicity": []}
        for i, m in enumerate(mols):
            smiles_csv["smiles"].append(
                Chem.MolToSmiles(Chem.RemoveHs(m))
            )
            smiles_csv["mutagenicity"].append(
                1 if m.GetProp("Mutagenicity") == "mutagen" else 0
            )
        df = pd.DataFrame(smiles_csv)
        df.to_csv("smiles_mutagenicity_raw.csv", index=False)
        self.smiles_data_raw = df

    def get_smiles_single_mols_no_ions(self):
        mols = [self.gen.get_mol(d) for d in self.data]
        smiles_csv = {"smiles": [], "mutagenicity": []}
        for i, m in enumerate(mols):
            if i not in self.faulty_mol_idxs:
                smiles_csv["smiles"].append(
                    Chem.MolToSmiles(Chem.RemoveHs(m))
                )
                smiles_csv["mutagenicity"].append(
                    1 if m.GetProp("Mutagenicity") == "mutagen" else 0
                )
        df = pd.DataFrame(smiles_csv)
        df.to_csv("smiles_single_mols_no_ions.csv", index=False)
        self.smiles_single_mols_no_ions = df

    def get_smiles_manual_filter(self):
        mols = [self.gen.get_mol(d) for d in self.data]
        smiles_csv = {"smiles": [], "mutagenicity": []}
        removed_csv = {"smiles": [], "mutagenicity": []}

        completely_removed = {"i": [], "mol": [], 'mutagenicity': []}
        all_mols_except_removed = {"i": [], "mol": [], 'mutagenicity': []}

        for i, m in enumerate(mols):
            if i not in self.faulty_mol_idxs:
                all_mols_except_removed['i'].append(i)
                all_mols_except_removed['mol'].append(Chem.RemoveHs(m))
                all_mols_except_removed['mutagenicity'].append(
                    1 if m.GetProp("Mutagenicity") == "mutagen" else 0
                )

            elif self._manual_fix(i):
                all_mols_except_removed['i'].append(i)
                all_mols_except_removed['mol'].append(self._manual_fix(i))
                all_mols_except_removed['mutagenicity'].append(
                    1 if m.GetProp("Mutagenicity") == "mutagen" else 0
                )

            elif self._manual_removal(i):
                completely_removed['i'].append(i)
                completely_removed['mol'].append(Chem.RemoveHs(m))
                completely_removed['mutagenicity'].append(
                    1 if m.GetProp("Mutagenicity") == "mutagen" else 0
                )

            elif self._rm_fragments(m):
                all_mols_except_removed['i'].append(i)
                all_mols_except_removed['mol'].append(self._rm_fragments(m))
                all_mols_except_removed['mutagenicity'].append(
                    1 if m.GetProp("Mutagenicity") == "mutagen" else 0
                )
            else:
                assert False, f"Unexpected mol {i}, {Chem.MolToSmiles(m)}"
        for i, m, mutagenicity in zip(all_mols_except_removed['i'],
                                      all_mols_except_removed['mol'],
                                      all_mols_except_removed['mutagenicity']):
            if Chem.MolToSmiles(m) not in smiles_csv["smiles"]:
                smiles_csv["smiles"].append(Chem.MolToSmiles(m))
                smiles_csv["mutagenicity"].append(mutagenicity)
            else:
                removed_csv["smiles"].append(Chem.MolToSmiles(m))
                removed_csv["mutagenicity"].append(mutagenicity)
        for i, m, mutagenicity in zip(completely_removed['i'],
                                      completely_removed['mol'],
                                      completely_removed['mutagenicity']):
            removed_csv["smiles"].append(Chem.MolToSmiles(m))
            removed_csv["mutagenicity"].append(mutagenicity)

        df = pd.DataFrame(smiles_csv)
        df.to_csv("smiles_mutagenicity_curated.csv", index=False)
        df = pd.DataFrame(removed_csv)
        df.to_csv("smiles_mutagenicity_removed.csv", index=False)

    def _manual_removal(self, idx):
        if idx == 985:  # mixture of methyl-phenols
            return True
        if idx == 3348:  # mixture: xylene
            return True
        if idx == 3968:  # mixture of ortho- and para-ethenetoluene
            return True
        if idx == 3975:  # mixture of chlorinated and non-chlorinated compound
            return True
        if idx == 4306:  # mixture of P(=O)-S-C and P(=S)-O-C compounds
            return True
        if idx == 2079:  # mixture of methylated/nonmethylated compounds
            return True
        if idx == 3571:  # mixture unclear compound
            return True
        if idx == 4010:  # mixture unclear compound
            return True
        if idx == 1056:  # unclear compound
            return True
        if idx == 1804:  # unclear compound
            return True
        if idx == 1528:  # sodium carbonate
            return True
        if idx == 2234:  # calcium carbonate
            return True
        if idx == 2482:  # potassium carbonate
            return True
        if idx == 277:  # ammonium hydrogencarbonate
            return True
        if idx == 518:  # ammonium carbonate
            return True
        if idx == 93:  # duplicate (different counter ions)
            return True
        if idx == 4188:  # duplicate (different counter ions)
            return True
        return None

    def _manual_fix(self, idx):
        # charged diazo groups
        if idx == 150:
            return Chem.MolFromSmiles('CCCCN(CC(O)C1=CC(=[N+]=[N-])C(=O)C=C1)N=O')
        if idx == 880:
            return Chem.MolFromSmiles('[N-]=[N+]=C1C=CC(=O)C=C1')
        if idx == 1260:
            return Chem.MolFromSmiles('CCOC(=O)CNC(=O)C=[N+]=[N-]')
        if idx == 1998:
            return Chem.MolFromSmiles('[N-]=[N+]=CC(=O)NCC(N)=O')
        if idx == 2115:
            return Chem.MolFromSmiles('[N-]=[N+]=CC(=O)OCC(N)C(=O)O')
        if idx == 3747:
            return Chem.MolFromSmiles('[N-]=[N+]=CC(=O)NCC(=O)NN')
        if idx == 4034:
            return Chem.MolFromSmiles('[N-]=[N+]=C1C=NC(=O)NC1=O')
        # zwitter ions
        if idx == 404:
            return Chem.MolFromSmiles('C[N+](C)(C)CC(=O)[O-]')
        if idx == 2701:
            return Chem.MolFromSmiles('C[n+]1cccc(C(=O)[O-])c1')
        if idx == 3051:
            return Chem.MolFromSmiles('CCN(CC)c1ccc2c(-c3ccc(S(=O)(=O)[O-])cc3S(=O)(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1')
        if idx == 3554:
            return Chem.MolFromSmiles(
                'CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2c(O)c(S(=O)(=O)[O-])cc3cc(S(=O)(=O)O)ccc23)cc1')
        if idx == 497:
            return Chem.MolFromSmiles(
                'CCN(Cc1cccc(S(=O)(=O)[O-])c1)c1ccc(C(=C2C=CC(=[N+](CC)Cc3cccc(S(=O)(=O)O)c3)C=C2)c2ccc(N(C)C)cc2)cc1')
        if idx == 3448:
            return Chem.MolFromSmiles('C[N+](C)(C)NCCC(=O)[O-]')
        # missing double bond
        if idx == 2564:
            return Chem.MolFromSmiles('Cn1cnc2c1nc[n+]([O-])C=2N')
        # remove ammonia
        if idx == 1674:
            return Chem.MolFromSmiles('NC(CCC(=O)O)C(=O)O')
        # two pos nitrogen neighbors also make zwitter ion
        if idx == 1153:
            return Chem.MolFromSmiles('C=C(C)C([O-])=N[N+](C)(C)CC(C)O')
        if idx == 3448:
            return Chem.MolFromSmiles('C[N+](C)(C)NCCC(=O)[O-]')
        # large fragments
        if idx == 1233:
            return Chem.MolFromSmiles(
                'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2nc(C(=O)NCCC[S+](C)C)cs2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1')
        # ionic
        if idx == 490:
            return Chem.MolFromSmiles('Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1')
        if idx == 638:
            return Chem.MolFromSmiles('Cn1c(-c2ccccc2)c(N=Nc2scc[n+]2C)c2ccccc21')
        if idx == 1125:
            return Chem.MolFromSmiles('CCN(CC)c1ccc2nc3ccc(N(CC)CC)cc3[o+]c2c1')
        if idx == 1151:
            return Chem.MolFromSmiles('CC(C)[N+](C)(CCOC(=O)C1c2ccccc2Oc2ccccc21)C(C)C')
        if idx == 1369:
            return Chem.MolFromSmiles('CC[n+]1c(-c2ccccc2)c2cc(N)ccc2c2ccc(N)cc21')
        if idx == 1379:
            return Chem.MolFromSmiles('ClC=CC[N+]12CN3CN(CN(C3)C1)C2')
        if idx == 1484:
            return Chem.MolFromSmiles('Cc1cc2nc3cc(C)c(N)cc3[n+](-c3ccccc3)c2cc1N')
        if idx == 1508:
            return Chem.MolFromSmiles('O=[N+]([O-])c1ccc(-n2nc(-c3ccccc3)n[n+]2-c2ccc(I)cc2)cc1')
        if idx == 1743:
            return Chem.MolFromSmiles('[O-][N+]([O-])=C1CCCC1')
        if idx == 1845:
            return Chem.MolFromSmiles('C[n+]1ccc(-c2cc[n+](C)cc2)cc1')
        if idx == 2972:
            return Chem.MolFromSmiles('C[n+]1cccc2c1C=CC1OC21')
        if idx == 3108:
            return Chem.MolFromSmiles('N#[N+]c1ccc([N+](=O)[O-])cc1')
        if idx == 3376:
            return Chem.MolFromSmiles('CC(C)=[N+]([O-])[O-]')
        if idx == 3387:
            return Chem.MolFromSmiles('C[n+]1c2cc(N)ccc2cc2ccc(N)cc21')
        if idx == 3408:
            return Chem.MolFromSmiles('C[N+](C)(C)CCNCCc1ccc(N=Nc2ccc([N+](=O)[O-])cc2Cl)cc1')
        if idx == 3484:
            return Chem.MolFromSmiles('CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(N(CC)CC)cc3[o+]c2c1')
        if idx == 3566:
            return Chem.MolFromSmiles('CCN(CC)c1ccc(C(=C2C=CC(=[N+](CC)CC)C=C2)c2ccc(N(CC)CC)cc2)cc1')
        if idx == 3908:
            return Chem.MolFromSmiles('C[n+]1ccc(-c2ccccc2)cc1')
        if idx == 3946:
            return Chem.MolFromSmiles('CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1')
        # large counter ions
        if idx == 58:
            return Chem.MolFromSmiles('O=C(O)COc1ccc(Cl)cc1Cl')
        if idx == 138:
            return Chem.MolFromSmiles('CN1CCN(C2=Nc3ccccc3Oc3ccc(Cl)cc32)CC1')
        if idx == 144:
            return Chem.MolFromSmiles('CC(C)(C)NCC(O)COc1nsnc1N1CCOCC1')
        if idx == 148:
            return Chem.MolFromSmiles('CNC(=O)Oc1ccc2c(c1)C1(C)CCN(C)C1N2C')
        if idx == 182:
            return Chem.MolFromSmiles('O=c1cn[nH]c(=O)[nH]1')
        if idx == 219:  # also ionic
            return Chem.MolFromSmiles('CCCCCCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1')
        if idx == 386:  # also ionic
            return Chem.MolFromSmiles('NC(=O)c1cc[n+](COC[n+]2ccccc2C=NO)cc1')
        if idx == 464:
            return Chem.MolFromSmiles('NNc1nnc(NN)c2ccccc12')
        if idx == 478:
            return Chem.MolFromSmiles('CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12')
        if idx == 560:
            return Chem.MolFromSmiles('CS(=O)(=O)OCCCNCCCOS(C)(=O)=O')
        if idx == 567:
            return Chem.MolFromSmiles('CN1Cc2c(N)cccc2C(c2ccccc2)C1')
        if idx == 640:
            return Chem.MolFromSmiles('COc1ccc(N)cc1N')
        if idx == 667:  # also ionic
            return Chem.MolFromSmiles('CCOCC(O)COc1ccc(NC(=O)CC[S+](C)C)cc1')
        if idx == 872:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2nc2ccccc21')
        if idx == 902:
            return Chem.MolFromSmiles('OCc1c2ccccc2cc2c1ccc1ccccc12')
        if idx == 1019:  # also ionic
            return Chem.MolFromSmiles('Cc1ccc([N+]#N)cc1')
        if idx == 1052:
            return Chem.MolFromSmiles('CNC(C)C(O)c1ccccc1')
        if idx == 1087:  # also ionic
            return Chem.MolFromSmiles('CN(C)C(=O)Oc1ccc[n+](C)c1')
        if idx == 1233:  # also ionic
            return Chem.MolFromSmiles(
                'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2nc(C(=O)NCCC[S+](C)C)cs2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1')
        if idx == 1307:  # also ionic
            return Chem.MolFromSmiles('CCCCCCCCCCSCn1cc[n+](C)c1')
        if idx == 1375:
            return Chem.MolFromSmiles('COc1cc(NC(C)CCCN)c2ncccc2c1')
        if idx == 1525:
            return Chem.MolFromSmiles(
                'CN1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)C(Cc4ccccc4)N3C2=O)C=C2c3cccc4[nH]cc(c34)CC21')
        if idx == 1587:  # also ionic
            return Chem.MolFromSmiles('CC[N+](CC)=c1ccc2nc3c(cc(N)c4ccccc43)oc-2c1')
        if idx == 1626:
            return Chem.MolFromSmiles('C1CCC(C(CC2CCCCN2)C2CCCCC2)CC1')
        if idx == 1654:
            return Chem.MolFromSmiles('CN(C)CCC(c1ccccc1)c1ccccn1')
        if idx == 1691:
            return Chem.MolFromSmiles('COc1ccc(CN(CCN(C)C)c2ccccn2)cc1')
        if idx == 1777:
            return Chem.MolFromSmiles('O=c1cn[nH]c(=O)[nH]1')
        if idx == 1949:
            return Chem.MolFromSmiles('CC(=C(OCCOc1ccc(Cl)cc1)c1ccc(Cl)cc1Cl)n1ccnc1')
        if idx == 2095:  # also ionic
            return Chem.MolFromSmiles('Oc1cc(O)c2cc(O)c(-c3cc(O)c(O)c(O)c3)[o+]c2c1')
        if idx == 2330:
            return Chem.MolFromSmiles('CCOc1ccc2nc3cc(N)ccc3c(N)c2c1')
        if idx == 2360:
            return Chem.MolFromSmiles(
                'CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O')
        if idx == 2371:
            return Chem.MolFromSmiles('Oc1cccc2cccnc12')
        if idx == 2485:  # also ionic
            return Chem.MolFromSmiles('C[N+](C)(C)Cc1ccccc1')
        if idx == 2513:  # also ionic
            return Chem.MolFromSmiles('CCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1')
        if idx == 2639:
            return Chem.MolFromSmiles('CCN(CC)CCNc1ccc(CO)c2sc3ccccc3c(=O)c12')
        if idx == 2694:
            return Chem.MolFromSmiles('Cc1cc(N)ccc1N')
        if idx == 2762:
            return Chem.MolFromSmiles('O=c1cn[nH]c(=O)[nH]1')
        if idx == 2777:  # also ionic
            return Chem.MolFromSmiles('CC[n+]1c2ccccc2nc2ccccc21')
        if idx == 2797:
            return Chem.MolFromSmiles('CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1')
        if idx == 2868:
            return Chem.MolFromSmiles('CCCn1cc2c3c(cccc31)C1C=C(C)CN(C)C1C2')
        if idx == 2956:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2c(N)c2ccccc21')
        if idx == 3020:
            return Chem.MolFromSmiles('Nc1ccc2cc3ccc(N)cc3nc2c1')
        if idx == 3047:
            return Chem.MolFromSmiles('N=c1ccc2nc3c(cc(N)c4ccccc43)oc-2c1')
        if idx == 3071:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2cc2ccccc21')
        if idx == 3081:
            return Chem.MolFromSmiles('Clc1ccc(COC(Cn2ccnc2)c2ccc(Cl)cc2Cl)cc1')
        if idx == 3135:  # also ionic
            return Chem.MolFromSmiles('CCN(CC)c1ccc(C(=C2C=CC(=[N+](CC)CC)C=C2)c2ccccc2)cc1')
        if idx == 3217:
            return Chem.MolFromSmiles(
                'CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C)C3C(O)(C(=O)OC)C(OC(C)=O)C5(CC)C=CCN6CCC43C65)C2)C1')
        if idx == 3279:
            return Chem.MolFromSmiles('O=P(O)(NCCCO)N(CCCl)CCCl')
        if idx == 3294:
            return Chem.MolFromSmiles('CCC(=O)N(c1ccc(Cl)c(Cl)c1)C1CCCC1N(C)C')
        if idx == 3340:
            return Chem.MolFromSmiles('CN(C)CCOC(C)(c1ccccc1)c1ccccn1')
        if idx == 3362:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2cc2c(N)cccc21')
        if idx == 3371:
            return Chem.MolFromSmiles(
                'CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C=O)C3C(O)(C(=O)OC)C(OC(C)=O)C5(CC)C=CCN6CCC43C65)C2)C1')
        if idx == 3424:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2cc2cc(N)ccc21')
        if idx == 3494:  # also ionic
            return Chem.MolFromSmiles('N#[N+]c1cccc2c1C(=O)c1ccccc1C2=O')
        if idx == 3503:  # also ionic
            return Chem.MolFromSmiles('C[N+]1=CC=C(c2ccccc2)CC1')
        if idx == 3594:
            return Chem.MolFromSmiles('N=C(N)c1ccc(OCCCCCOc2ccc(C(=N)N)cc2)cc1')
        if idx == 3719:  # also ionic
            return Chem.MolFromSmiles('C[n+]1c2ccccc2cc2ccc(N)cc21')
        if idx == 3860:
            return Chem.MolFromSmiles('CNc1ccc(O)cc1')
        if idx == 3919:
            return Chem.MolFromSmiles('COc1ccc2c3c1OC1C(O)C=CC4C(C2)N(C)CCC341')
        if idx == 4054:
            return Chem.MolFromSmiles('O=C(O)COc1ccc(Cl)cc1Cl')
        if idx == 4239:  # also ionic
            return Chem.MolFromSmiles('C[N+]1(C)CCOCC1')
        if idx == 4317:
            return Chem.MolFromSmiles('NCC(=O)NCC(=O)NCC(=O)NCC(=O)O')
        # change compounds with [SH] to [S+]
        if idx == 278:
            return Chem.MolFromSmiles(
                'Cc1c(N)nc(C(CC(N)=O)NCC(N)C(N)=O)nc1C(=O)NC(C(=O)NC(C)C(O)C(C)C(=O)NC(C(=O)NCCc1nc(-c2ncc(C(=O)NCCC[S+](C)C)s2)cs1)C(C)O)C(OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1c[nH]cn1')
        if idx == 2511:
            return Chem.MolFromSmiles('Cc1cc2c(cc1N)=[S+]c1cc(N(C)C)ccc1N=2')

        return None

    def _rm_fragments(self, mol):
        smi = Chem.MolToSmiles(mol)
        frags = smi.split(".")
        keep_frags = [f for f in frags if len(f) > 7]
        if len(keep_frags) == 1:
            return Chem.MolFromSmiles(keep_frags[0])
        else:
            return None

    def view_data_on_idx(self, idx):
        print("faulty_idx:", idx)
        for atom in self.mols[idx].GetAtoms():
            atom.SetProp('atomNote', str(atom.GetIdx()))
        display(Chem.Draw.MolToImage(self.mols[idx], size=(600, 600)))
        return self.mols[idx]


if __name__ == "__main__":
    data = Mutagenicity(root='',mode='training')

    mf = ManualFixer(data)
    mf.get_smiles_data_raw()
    mf.get_smiles_single_mols_no_ions()
    mf.get_smiles_manual_filter()