# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graph neural network (GNN) explainability research project focused on molecular graph analysis, particularly for the Mutagenicity dataset. The codebase implements counterfactual explanation methods using variational graph autoencoders (VGAE) and graph edit distance (GED) calculations.

## Core Architecture

### Dataset Pipeline
- **Datasets** (`datasets/`): Custom PyTorch Geometric dataset implementations for multiple graph datasets (BA3Motif, Mutagenicity/MUTAG, NCI1, BBBP, Synthetic, WebDataset)
- **Data Loading**: Uses `utils.dataset.get_datasets()` to load train/val/test splits
- **Molecular Encoding**: Atoms are one-hot encoded using `MUTAG_atom_map` (14 atom types: C, O, Cl, H, N, F, Br, S, P, I, Na, K, Li, Ca), edges are one-hot encoded (3 bond types: single, double, triple)

### GNN Models
- **Model Zoo** (`gnns/`): Dataset-specific GCN architectures (Mutag_GCN, BA3MotifNet, NCI1GCN, etc.)
- **Loading**: Pre-trained models stored in `param/gnns/{dataset}_gcn.pt`
- **Key Methods**: `get_pred()` returns predictions, `get_graph_rep()` returns graph embeddings

### Explainability Pipeline

#### 1. Subgraph Extraction (`utils/ps/mol_bpe.py`)
- Uses **graph byte-pair encoding (BPE)** to extract frequent molecular subgraphs from SMILES
- `graph_bpe()`: Iteratively merges frequent subgraphs to build vocabulary
- `Tokenizer`: Converts molecules to subgraph representations
- Principal subgraphs are ranked by frequency and stored

#### 2. Graph Pair Construction (`utils/pair_data.py`)
- `GraphPairData`: Creates training pairs of (original graph, target graph)
- For each graph, finds top-k most similar graphs with different predicted labels using cosine similarity on graph embeddings
- Filters by prediction confidence threshold (default 0.9)
- Returns paired graphs with their embeddings and distances

#### 3. Counterfactual Generation (`models/`)
- **MyExplainer** (`myexplainer.py`): VAE-based model with encoder-decoder architecture
  - Encoder: Takes graph features → latent representation
  - Decoder: Latent + counterfactual label → reconstructed graph
  - Uses dense graph operations (DenseGCNConv/DenseGATConv)
- **VGAE variants** (`vgae.py`, `vgae_v2.py`, `vgae_v3.py`): Different VGAE implementations for graph transformation

#### 4. Subgraph Matching (`utils/subgraph_utils.py`)
- `find_largest_subgraph()`: Finds the largest extracted subgraph present in a target molecule using RDKit substructure matching
- `generate_subgraph_mask()`: Creates node/edge masks for identified subgraphs
- Uses SMARTS/SMILES to RDKit Mol conversions with hydrogen handling and valence correction

### Graph Edit Distance (`GED/`)
- **GRAPHEDX** (`graphedx.py`): Neural GED estimation using Sinkhorn iterations
- Two-level transport plan: node alignment → edge alignment
- Used for computing graph similarity metrics

### Utilities
- **graph_utils.py**: Conversions between PyG Data ↔ RDKit Mol ↔ SMILES/SMARTS
  - `data_to_mol()`: PyG Data → RDKit with atom/bond mapping
  - `smarts_to_data()`: SMARTS → PyG Data with one-hot encoding
  - `_sanitize_with_valence_correction()`: Handles valence issues in generated molecules
- **vis_utils.py**: Visualization of subgraphs with edge masks
- **train_utils.py**: Training/testing utilities (`Gtrain`, `Gtest`)

## Main Workflow

The typical execution flow in `main.py`:

1. **Load dataset and GNN**: `get_datasets()` + load pretrained GNN from `param/gnns/`
2. **Extract subgraph vocabulary**: Use `graph_bpe()` to get frequent molecular patterns from SMILES (stores lists like `smis_0`, `smis_1` for each class)
3. **Find explanatory subgraphs**: For a target molecule, use `find_largest_subgraph()` to identify which vocabulary subgraph is present
4. **Generate masks**: Use `generate_subgraph_mask()` to get node/edge masks for visualization
5. **Visualize**: `visualize_subgraph()` highlights the explanatory subgraph

## Development Commands

### Running the Main Pipeline
```bash
python main.py --cuda 0 --dataset mutag --top_k 5 --threshold 0.9 --epochs 200
```

**Key Arguments:**
- `--cuda`: GPU device ID
- `--dataset`: Dataset name (MUTAG, etc.)
- `--gnn_path`: Directory for pretrained GNNs (default: `param/`)
- `--top_k`: Number of similar graphs for pairing
- `--threshold`: Prediction confidence threshold
- `--vocab_len`: Number of subgraphs to extract
- `--epochs`: Training epochs for VGAE

### Working with the Codebase

**Important Constants:**
- Atom mapping: 14 types in `MUTAG_atom_map` (graph_utils.py, subgraph_utils.py)
- Bond mapping: 3 types - {0: SINGLE, 1: DOUBLE, 2: TRIPLE}
- Device: Configured via args.device = `torch.device(f'cuda:{args.cuda}')`

**Data Formats:**
- Graphs use PyTorch Geometric `Data` objects with `.x` (node features), `.edge_index`, `.edge_attr`, `.y` (labels)
- Molecular structures interchange between PyG Data, RDKit Mol, SMILES, and SMARTS representations

**Pre-trained Model Loading:**
```python
gnn = torch.load(f'param/gnns/{dataset_name}_gcn.pt', map_location=device)
```

## Known Patterns

1. **SMILES/SMARTS Processing**: Always uses RDKit with explicit hydrogen handling and valence correction via `_sanitize_with_valence_correction()`

2. **Graph Pairing**: The paired dataset construction is computationally expensive - uses batched operations with cosine similarity for efficiency

3. **Subgraph Vocabulary**: The BPE extraction runs in multiprocessing mode (see `mol_bpe.py`) and generates sorted vocabularies by frequency

4. **Visualization**: Subgraph masks are boolean tensors aligned with node/edge indices - edge masks handle bidirectional edges in undirected graphs
