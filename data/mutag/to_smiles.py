import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
import os
from datasets.mutag_dataset import Mutagenicity
from smiles import to_smiles

if __name__ == "__main__":

    # Load a PyG dataset (replace with your dataset)
    dataset = Mutagenicity(root='',mode='testing')
    for data in dataset:
        # print(data)
        # print(data.edge_index)
        if len(dataset)==3337:
            print("Dataset length is correct.")
            if to_smiles(data)=='Cn1cnc2c(N)[n+]([O-])cnc21':
                print("Found it!")
                break

        # print(data.name+":"+to_smiles(data))
        # print("non-mutagenic" if data.y.item() else "mutagenic")
        # print("========")



        # break

    # Convert and save to SMILES
    # convert_dataset_to_smiles(dataset, output_file="qm9_smiles.txt")