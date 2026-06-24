import argparse
import copy
import os
import os.path as osp
import random
import sys
import time

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm

from datasets.alkane_carbonyl_dataset import AlkaneCarbonyl
from utils import Gtest, Gtrain, set_seed

try:
    from .model_utils import (
        EdgeWeightedGATConv,
        get_class_weights,
        get_pool_output_dim,
        pool_graph_representation,
        validate_graph_pooling,
    )
except ImportError:
    from gnns.model_utils import (
        EdgeWeightedGATConv,
        get_class_weights,
        get_pool_output_dim,
        pool_graph_representation,
        validate_graph_pooling,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Alkane-Carbonyl Model")

    parser.add_argument(
        "--data_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "data", "alkane_carbonyl"),
        help="Input data path.",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="Path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=3, help="Number of convolution layers.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["mean", "mean_max"],
        help="Graph readout for Alkane-Carbonyl training. 'mean' is the new default; "
             "'mean_max' preserves the previous behavior.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=40,
        help="Stop training if validation loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement required to refresh the best checkpoint.",
    )
    parser.add_argument(
        "--random_label",
        type=bool,
        default=False,
        help="Train a model under label randomization for sanity check.",
    )
    return parser.parse_args()


from gnns.base import BaseGNNClassifier


class AlkaneCarbonylGCN(BaseGNNClassifier):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        conv_unit=3,
        dropout=0.3,
        graph_pooling="mean_max",
    ):
        super().__init__(in_channels, hidden_dim=hidden_dim, num_classes=2)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList([ReLU() for _ in range(conv_unit)])
        self.dropout = dropout
        self.graph_pooling = graph_pooling

        validate_graph_pooling(self.graph_pooling)

        self.convs.append(
            EdgeWeightedGATConv(
                in_channels=in_channels,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                dropout=dropout,
            )
        )
        for _ in range(conv_unit - 2):
            self.convs.append(
                EdgeWeightedGATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout,
                )
            )
        self.convs.append(
            EdgeWeightedGATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                dropout=dropout,
            )
        )

        # Each convolution layer needs its own normalization statistics.
        self.batch_norms.extend([BatchNorm(hidden_dim) for _ in range(conv_unit)])
        pool_out_dim = get_pool_output_dim(hidden_dim, self.graph_pooling)
        self.ffn = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim),
            ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.softmax = Softmax(dim=1)

    def _forward_convs(self, x, edge_index, edge_weight=None):
        for layer_idx, (conv, batch_norm, relu) in enumerate(zip(self.convs, self.batch_norms, self.relus)):
            residual = x
            x = conv(x, edge_index, edge_weight)
            x = batch_norm(x)
            x = relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if layer_idx > 0:
                x = x + residual
        return x

    def _pool(self, node_emb, batch):
        return pool_graph_representation(node_emb, batch, getattr(self, "graph_pooling", "mean_max"))

    def _classify(self, graph_emb):
        return self.ffn(graph_emb)

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


if __name__ == "__main__":
    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    test_dataset = AlkaneCarbonyl(args.data_path, mode="testing")
    val_dataset = AlkaneCarbonyl(args.data_path, mode="evaluation")
    train_dataset = AlkaneCarbonyl(args.data_path, mode="training")

    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for graph in dataset:
                graph.y.fill_(random.choice([0, 1]))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = AlkaneCarbonylGCN(
        in_channels=train_dataset[0].x.size(1),
        hidden_dim=args.hidden_dim,
        conv_unit=args.num_unit,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
    ).to(device)

    class_counts, class_weights = get_class_weights(train_dataset)
    print(
        "Training label distribution: class 0 = {}, class 1 = {}, class weights = [{:.4f}, {:.4f}]".format(
            int(class_counts[0]),
            int(class_counts[1]),
            float(class_weights[0]),
            float(class_weights[1]),
        )
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=10,
        min_lr=1e-5,
    )

    best_epoch = 0
    best_val_error = float("inf")
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = float(Gtrain(train_loader, model, optimizer, device=device, criterion=criterion))
        _, train_acc = Gtest(train_loader, model, device=device, criterion=criterion)
        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=criterion)
        train_acc = float(train_acc)
        val_error = float(val_error)
        val_acc = float(val_acc)
        scheduler.step(val_error)

        improved = (best_val_error - val_error) > args.min_delta
        if not improved and abs(val_error - best_val_error) <= args.min_delta:
            improved = val_acc > best_val_acc

        if improved:
            best_epoch = epoch
            best_val_error = val_error
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        t2 = time.time()
        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader, model, device=device, criterion=criterion)
            test_error = float(test_error)
            test_acc = float(test_acc)
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, "
                "Test acc: {:.5f}".format(epoch, t2 - t1, lr, loss, test_error, test_acc)
            )
        else:
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, "
                "Validation Loss: {:.5f}, Validation acc: {:.5f}".format(
                    epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc
                )
            )

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                "Early stopping at epoch {:d}. Best validation loss {:.5f}, "
                "best validation acc {:.5f} at epoch {:d}.".format(
                    epoch,
                    best_val_error,
                    best_val_acc,
                    best_epoch,
                )
            )
            break

    model.load_state_dict(best_state)
    test_error, test_acc = Gtest(test_loader, model, device=device, criterion=criterion)
    test_error = float(test_error)
    test_acc = float(test_acc)
    print(
        "Best checkpoint from epoch {:d}: Validation Loss {:.5f}, Validation acc {:.5f}, "
        "Final Test Loss {:.5f}, Final Test acc {:.5f}".format(
            best_epoch,
            best_val_error,
            best_val_acc,
            test_error,
            test_acc,
        )
    )

    save_path = "alkane_carbonyl_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
