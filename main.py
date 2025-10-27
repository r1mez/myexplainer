import argparse
import sys

import networkx as nx
from torch_geometric.utils import to_networkx

sys.path.append("..")

import torch
import time
from utils import get_datasets,extract_data, GraphPairData, custom_collate_fn, GraphTrainData
from torch_geometric.loader import DataLoader
from torch.nn.functional import cosine_similarity
from gnns import Mutag_GCN
from datasets import Mutagenicity
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils.ps.mol_bpe import graph_bpe
from utils.subgraph_utils import find_largest_subgraph, generate_subgraph_mask
from utils.vis_utils import visualize_subgraph
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=2, help="GPU device.")
    parser.add_argument("--gnn_path", type=str, default="param/", help="GNN directory.")
    parser.add_argument("--dataset", type=str, default="mutag", choices=['mutag'], help="Dataset name.")
    parser.add_argument("--similarity", type=str, default="cosine", choices = ["cosine"],help="Metric for graph similarity calculation.")
    parser.add_argument("--top_k", type=int, default=5, help="number of top similar graphs for model training.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for data extraction.")
    parser.add_argument("--task", type=str, default='gc', choices = ['gc','nc'], help="Random seed for reproducibility.")
    parser.add_argument("--label", type=int, default=1, help="Label to be explained.")
    parser.add_argument("--vocab_len", type=int, default=50, help="Number of extracted subgraphs")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    return parser.parse_args()

args = parse_args()
args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset.lower())

gnn = torch.load(f'param/gnns/{args.dataset.lower()}_gcn.pt', map_location=args.device)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
# smiles_list_0=[]
# smiles_list_1=[]
# for data in train_loader:
#     data.to(args.device)
#     # 获取预测结果并确定每个样本的类别
#     pred_labels = gnn.get_pred(data.x, data.edge_index, data.batch)[0].argmax(dim=1)
#
#     # 分别处理每个样本
#     for i in range(len(pred_labels)):
#         if pred_labels[i] == 0:
#             smiles_list_0.append(data.smiles[i] if isinstance(data.smiles, list) else data.smiles)
#         else:
#             smiles_list_1.append(data.smiles[i] if isinstance(data.smiles, list) else data.smiles)
#
# f=open('data/mutag/smiles/smiles_0.txt',"w")
# for i in smiles_list_0:
#     f.write(i+'\n')
# f.close()
#
# f=open('data/mutag/smiles/smiles_1.txt',"w")
# for i in smiles_list_1:
#     f.write(i+'\n')
# f.close()




vocab_len = 1000
smis_0,_ = graph_bpe('data/mutag/smiles/smilesf_0.txt', vocab_len, f'data/mutag/smiles_bpe_{vocab_len}_0.txt', 16, False)
smis_1,_ = graph_bpe('data/mutag/smiles/smiles_1.txt', vocab_len, f'data/mutag/smiles_bpe_{vocab_len}_1.txt', 16, False)

# smis_0 = ['C', 'cc', 'cccc', 'O', 'N', 'CC', 'c1ccccc1', 'CCO', 'Cc1ccccc1', '[N+][O-]', 'O=[N+][O-]', 'CCN', 'cn', 'Cl', 'cccccccc', 'CO', 'Nc1ccccc1', 'CN', 'ccc1ccccc1', 'S', 'c1ccc2ccccc2c1', 'O=[N+]([O-])c1ccccc1', 'O=Cc1ccccc1', 'ccn', 'O=S', 'CCl', 'CC=O', 'CCNC', 'Br', 'CC1CO1', 'ccccC', 'H', 'Oc1ccccc1', 'CCCl', 'cccc1ccccc1', 'c1ccc2ncccc2c1', 'cccn', 'CCC', 'CC(N)=O', 'N=O', 'C1CO1', 'CC(=O)O', 'OCCO', 'CNC', 'O=S=O', 'Cc1ccccc1N', 'ccccc1ccccc1', 'c1ccncc1', 'CCc1ccccc1', 'CNc1ccccc1', 'ncn', 'O=C1c2ccccc2C(=O)c2ccccc21', 'cccccc', 'NC=O', 'O=[SH](=O)O', 'Nc1ccc2ccccc2c1', 'F', 'O=CO', 'O=C1c2ccccc2C(=O)c2c(O)cccc21', 'CCOCCO', 'CCCC', 'cccccccc[N+](=O)[O-]', 'Cc1ccccc1[N+](=O)[O-]', 'c1ccc2nc3ccccc3cc2c1', 'c1ccsc1', 'c1ccc2c(c1)ccc1ccccc12', 'ClCCl', 'Cc1ccco1', 'CC(O)CO', 'Cncn', 'NO', 'CCOP', 'c1ccc2[nH]ccc2c1', 'Nc1ccc([N+](=O)[O-])cc1', 'c1cncn1', 'ccccccc1ccccc1', 'CCCO', 'Cc1ccc([N+](=O)[O-])o1', 'P', 'c1ccc2cc3ccccc3cc2c1', 'CCNC=O', 'ccnc', 'Nc1cccc2ccccc12', 'Cc1ccc([N+](=O)[O-])cc1', 'O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'CC(C)O', 'Cnc(n)N', 'CCNCC', 'c1ccc(c2ccccc2)cc1', 'c1ccc2occcc2c1', 'nc1ccccc1', 'ccccn', 'c1cncnc1', 'cncn', 'CCNCCN', 'cc1ccccc1', 'Cc1ccc(N)cc1', 'ccc', 'N=Nc1ccccc1', 'I']
# smis_1 = ['C', 'CC', 'cc', 'O', 'cccc', 'CCCC', 'N', 'c1ccccc1', 'CCC', 'CO', 'CCO', 'Cl', 'CN', 'cn', 'Cc1ccccc1', 'CCCCO', 'CCN', 'S', 'Oc1ccccc1', 'O=S', 'CCCCCCCC', 'O=CO', 'Clc1ccccc1', 'CCC=O', 'F', 'CC(=O)O', 'CCC(C)C', 'Br', 'O=S=O', 'CC=O', 'NC=O', 'Nc1ccccc1', 'CC(C)C', 'CCCC=O', 'cncn', 'CCCCC', 'ccn', 'H', 'CC(N)=O', 'O=Cc1ccccc1', 'CNC', 'CCCO', 'c1ccncc1', 'CCCC(=O)O', 'O=[SH](=O)O', 'P', 'CCC(C)O', 'OCCO', 'c[nH]', 'CCC(=O)O', 'COc1ccccc1', 'c1cncnc1', 'FCc1ccccc1', 'OP', 'ccccc', 'CCC(C)=O', 'Clc1cccc(Cl)c1', 'CC(O)CCO', 'O=C(O)c1ccccc1', 'Brc1ccccc1', 'OCC1OCC(O)CC1O', 'CCNCC', 'CC(C)O', 'CCNC', 'COP', 'CCCC(C)O', 'ccc1ccccc1', 'CC=CC', 'FC(F)c1ccccc1', 'NCCO', 'c1ccc2[nH]ccc2c1', 'Oc1ccccc1Br', 'c1ccc2occcc2c1', 'Oc1ccc(Cl)cc1', 'C[N+]', 'CCCC(C)C', 'CC(=O)Nc1ccccc1', 'c1ccc2ccccc2c1', 'N[SH](=O)=O', 'C=C(C)C=O', 'CC(O)CO', 'CCCCl', 'Cc1ccc(O)cc1', 'C1CCCCC1', 'CC(C)N', 'ccc', 'CCCCCC', 'CCCCC(=O)O', 'CCC(O)CO', 'c1ccc2ncccc2c1', 'CCC(N)=O', 'CC(=O)c1ccccc1', 'cccc1ccccc1', 'CC(C)=O', 'CCCCCC(C)CC', 'I', 'nc=O', '[N+]=O', 'O=[N+][O-]', 'CCc1ccccc1']

print("smis_0:",smis_0)
print("smis_1:",smis_1)
smis_0_set, smis_1_set = set(smis_0), set(smis_1)
print("smis_0_set-smis_1_set:",smis_0_set-smis_1_set)
print("smis_1_set-smis_0_set:",smis_1_set-smis_0_set)

# smis = ['C', 'cc', 'cccc', 'O', 'N', 'CC', 'c1ccccc1', 'CCO', 'Cc1ccccc1', '[N+][O-]', 'O=[N+][O-]', 'CCN', 'cn', 'Cl', 'cccccccc', 'CO', 'Nc1ccccc1', 'CN', 'ccc1ccccc1', 'S', 'c1ccc2ccccc2c1', 'O=[N+]([O-])c1ccccc1', 'O=Cc1ccccc1', 'ccn', 'O=S', 'CCl', 'CC=O', 'CCNC', 'Br', 'CC1CO1', 'ccccC', 'H', 'Oc1ccccc1', 'CCCl', 'cccc1ccccc1', 'c1ccc2ncccc2c1', 'cccn', 'CCC', 'CC(N)=O', 'N=O', 'C1CO1', 'CC(=O)O', 'OCCO', 'CNC', 'O=S=O', 'Cc1ccccc1N', 'ccccc1ccccc1', 'c1ccncc1', 'CCc1ccccc1', 'CNc1ccccc1', 'ncn', 'O=C1c2ccccc2C(=O)c2ccccc21', 'cccccc', 'NC=O', 'O=[SH](=O)O', 'Nc1ccc2ccccc2c1', 'F', 'O=CO', 'O=C1c2ccccc2C(=O)c2c(O)cccc21', 'CCOCCO', 'CCCC', 'cccccccc[N+](=O)[O-]', 'Cc1ccccc1[N+](=O)[O-]', 'c1ccc2nc3ccccc3cc2c1', 'c1ccsc1', 'c1ccc2c(c1)ccc1ccccc12', 'ClCCl', 'Cc1ccco1', 'CC(O)CO', 'Cncn', 'NO', 'CCOP', 'c1ccc2[nH]ccc2c1', 'Nc1ccc([N+](=O)[O-])cc1', 'c1cncn1', 'ccccccc1ccccc1', 'CCCO', 'Cc1ccc([N+](=O)[O-])o1', 'P', 'c1ccc2cc3ccccc3cc2c1', 'CCNC=O', 'ccnc', 'Nc1cccc2ccccc12', 'Cc1ccc([N+](=O)[O-])cc1', 'O=[N+]([O-])c1cccc([N+](=O)[O-])c1', 'CC(C)O', 'Cnc(n)N', 'CCNCC', 'c1ccc(c2ccccc2)cc1', 'c1ccc2occcc2c1', 'nc1ccccc1', 'ccccn', 'c1cncnc1', 'cncn', 'c1ccc2sncc2c1', 'CCNCCN', 'cc1ccccc1', 'Cc1ccc(N)cc1', 'ccc', 'N=Nc1ccccc1', 'ncc1ccccc1', 'cccc1cccc2ccccc12', 'COP', 'O=CCO', 'c1ccc2c(c1)Cc1ccccc1-2', 'c1ccc(C2CN2)cc1', 'CCNC(C)=O', 'CCBr', 'C=O', 'I']
# smis = ['C', 'cc', 'cccc', 'O', 'N', 'CC', 'c1ccccc1', 'CCO', 'Cc1ccccc1', '[N+][O-]', 'O=[N+][O-]', 'CCN', 'cn', 'Cl', 'cccccccc', 'CO', 'Nc1ccccc1', 'CN', 'ccc1ccccc1', 'S', 'c1ccc2ccccc2c1', 'O=[N+]([O-])c1ccccc1', 'O=Cc1ccccc1', 'ccn', 'O=S', 'CCl', 'CC=O', 'CCNC', 'Br', 'CC1CO1', 'ccccC', 'H', 'Oc1ccccc1', 'CCCl', 'cccc1ccccc1', 'c1ccc2ncccc2c1', 'cccn', 'CCC', 'CC(N)=O', 'N=O', 'C1CO1', 'CC(=O)O', 'OCCO', 'CNC', 'O=S=O', 'Cc1ccccc1N', 'ccccc1ccccc1', 'F', 'P', 'I']
list_0=[]
list_1=[]
for data in train_dataset:
    data.to(args.device)
    # 获取预测结果并确定每个样本的类别
    pred_labels = gnn.get_pred(data.x, data.edge_index, data.batch)[0].argmax(dim=1)

    # 分别处理每个样本
    for i in range(len(pred_labels)):
        if pred_labels[i] == 0:
            list_0.append(data)
        else:
            list_1.append(data)
# data = list_0[159]
data = test_dataset[250]
print(data.y)
# print(find_largest_subgraph(smis,data))
# print(data.smiles)
print(data.smiles + '.' + find_largest_subgraph(smis_0,data))
print("data.x",data.x)
print("data.edge_index",data.edge_index)
print(generate_subgraph_mask(find_largest_subgraph(smis_0,data),data))
edge_mask = generate_subgraph_mask(find_largest_subgraph(smis_0,data),data)[1]
# visualize_subgraph(data, edge_mask)
# visualize_subgraph(data, edge_mask[1])
vocab = {0:smis_0,1:smis_1}
train_dataset = GraphTrainData(args, train_loader, gnn, vocab)
print(train_dataset[250])
for i in range(119,201):
    visualize_subgraph(train_dataset[i]['graph'], train_dataset[i]['freq_sub_mask'])















