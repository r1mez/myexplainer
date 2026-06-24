import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

from config import ExplainerConfig
from evaluationV2 import evaluate
from models.myexplainerV2 import MyExplainerV2
from utils import get_datasets, train_collate_fn
from utils.dataset_registry import get_dataset_entry
from utils.pair_data import MappedDataset
from utils.subgraph_method import subgraph_mining
from utils.train_myexplainer import train_myexplainerV2

from gnns import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MyExplainer model')

    # 基础设置
    parser.add_argument('--cuda', type=int, default=2, help='GPU device')
    parser.add_argument('--dataset', type=str, default='mutag', help='Dataset name')
    parser.add_argument('--gnn_path', type=str, default='param/', help='GNN directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--train_mode',type=bool,default=True,help='Current mode')
    parser.add_argument('--task', type=str, default='graph', help='Task type: graph classification or node classification')


    # 数据参数
    parser.add_argument('--top_k', type=int, default=1, help='Number of similar graphs for pairing')
    parser.add_argument('--threshold', type=float, default=0, help='Prediction confidence threshold')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # 模型参数
    # ba2: 128,32    mutag:256,32
    # parser.add_argument('--x_dim', type=int, default=10, help='Node feature dimension (14 for mutag)')
    parser.add_argument('--h_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=32, help='Latent dimension')

    parser.add_argument('--max_num_nodes', type=int, default=25, help='Maximum number of nodes in a graph')         # 53, 28
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    parser.add_argument('--subgraph_method',type=str,default='genGraphEx',help='Subgraph method')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_dataset(config):
    """Load dataset splits and update config with feature dimensions."""
    print("\n1. Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(name=config.dataset.lower())
    x_dim = train_dataset[0].x.shape[1]
    edge_attr_dim = train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0
    config = config.with_dataset_dims(x_dim, edge_attr_dim)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, config


def load_gnn(config):
    """Load pre-trained GNN classifier and freeze its parameters."""
    print("\n2. Loading pre-trained GNN classifier...")
    entry = get_dataset_entry(config.dataset)
    gnn = torch.load(f'param/gnns/{entry["gnn_file"]}', map_location=config.device)
    for p in gnn.parameters():
        p.requires_grad_(False)
    gnn.eval()
    print("  GNN loaded successfully")
    return gnn


def split_by_prediction(gnn, dataset, device):
    """Split dataset by GNN predictions into class 0 and class 1 subsets.

    Returns the splits dict along with per-sample labels and probabilities
    (needed downstream for building mapped datasets).
    """
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = gnn(data.x, data.edge_index, data.batch)
            pred_probs.extend(out.softmax(dim=1))
            preds = out.argmax(dim=1).cpu()
            pred_labels.extend(preds)

    indices_0 = [i for i, pred in enumerate(pred_labels) if pred == 0]
    indices_1 = [i for i, pred in enumerate(pred_labels) if pred == 1]
    return {0: dataset[indices_0], 1: dataset[indices_1]}, pred_labels, pred_probs


def mine_subgraphs(config, splits):
    """Run frequent subgraph mining on class-split datasets."""
    return subgraph_mining(config, splits)


def build_mapped_datasets(config, train_dataset, val_dataset, test_dataset,
                          patterns, pred_labels, pred_probs, gnn):
    """Create VF2-mapped datasets and wrap them in DataLoaders."""
    print("\n4. Creating dataset with subgraph masks...")
    train_dataset_with_masks = MappedDataset(config, train_dataset, patterns, pred_labels, pred_probs)
    test_dataset_with_masks = MappedDataset(config, test_dataset, patterns, gnn=gnn)
    val_dataset_with_masks = MappedDataset(config, val_dataset, patterns, gnn=gnn)

    print("\n5. Creating masked data loader...")
    train_loader = TorchDataLoader(
        train_dataset_with_masks,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    test_loader = TorchDataLoader(
        test_dataset_with_masks,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    val_loader = TorchDataLoader(
        val_dataset_with_masks,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn
    )
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total batches: {len(train_loader)}")
    return train_loader, val_loader, test_loader


def train_explainer(config, model, gnn, train_loader, eval_loader):
    """Train explainer from scratch or load from checkpoint."""
    if config.train_mode:
        print("\n7. Initializing MyExplainer model...")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {num_params:,}")

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True,
            min_lr=config.scheduler_min_lr
        )
        print(f"  Optimizer: Adam")
        print(f"  Learning rate: {config.lr}")
        print(f"  Weight decay: {config.weight_decay}")

        print("\n8. Training MyExplainer with subgraph masks...")
        print("=" * 80)

        trained_model, losses = train_myexplainerV2(
            config=config,
            model=model,
            gnn=gnn,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config.epochs
        )
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    else:
        print("\n8. Loading Trained MyExplainer...")
        print("=" * 80)
        model.load_state_dict(torch.load(f'param/myexplainer_{config.dataset.lower()}_best.pt', map_location=config.device))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        trained_model = model
        print("\n" + "=" * 80)
        print("Loading completed successfully!")
        print("=" * 80)
    return trained_model


def _print_evaluation_results(evaluation_metrics):
    """Print evaluation metrics to stdout."""
    print("\nEvaluation Results on Validation Set:")
    print(
        "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
            evaluation_metrics["validity"],
            int(evaluation_metrics["successful"]),
            int(evaluation_metrics["total"]),
        )
    )
    print(
        "  Proximity ↓: {:.4f}".format(
            evaluation_metrics["proximity"]
        )
    )
    # print(
    #     "  Fidelity+ ↑: {:.4f}".format(
    #         evaluation_metrics["fidelity+"]
    #     )
    # )
    # print(
    #     "  Fidelity- ↓: {:.4f}".format(
    #         evaluation_metrics["fidelity-"]
    #     )
    # )
    print(
        "  Fidelity_prob ↑: {:.4f}".format(
            evaluation_metrics["fidelity"]
        )
    )
    print(
        "  Sparsity ↑: {:.4f}".format(
            evaluation_metrics["sparsity"]
        )
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = ExplainerConfig.from_args(args)
    print(f"Using device: {config.device}")

    train_dataset, val_dataset, test_dataset, config = load_dataset(config)
    gnn = load_gnn(config)
    splits, pred_labels, pred_probs = split_by_prediction(gnn, train_dataset, config.device)
    patterns = mine_subgraphs(config, splits)
    train_loader, val_loader, test_loader = build_mapped_datasets(
        config, train_dataset, val_dataset, test_dataset, patterns, pred_labels, pred_probs, gnn)
    model = MyExplainerV2(config, gnn).to(config.device)
    trained_model = train_explainer(config, model, gnn, train_loader, val_loader)

    print("\n9. Evaluating on validation set...")
    print("=" * 80)
    metrics = evaluate(config=config, model=trained_model, gnn=gnn, data_loader=val_loader)
    _print_evaluation_results(metrics)
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
