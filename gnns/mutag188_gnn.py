import argparse
import copy
import os
import os.path as osp
import random
import sys
import time

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, CrossEntropyLoss, Linear, ModuleList, ReLU, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from datasets import MUTAG188
from utils import Gtest, Gtrain, set_seed

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train MUTAG188 Model")

    parser.add_argument(
        "--data_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "data", "mutag188"),
        help="Input data path.",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "param", "gnns"),
        help="path for saving trained model.",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=3, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )

    return parser.parse_args()


class Mutag188_GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        num_unit: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_unit = num_unit
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.convs = ModuleList()
        self.bns = ModuleList()
        self.act = ReLU()

        for layer_idx in range(num_unit):
            in_c = in_channels if layer_idx == 0 else hidden_dim
            self.convs.append(GCNConv(in_c, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.softmax = Softmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_node_reps(self, x, edge_index, edge_weight=None):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = bn(h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def get_graph_rep(self, x, edge_index, batch, edge_weight=None):
        node_x = self.get_node_reps(x, edge_index, edge_weight=edge_weight)
        return global_mean_pool(node_x, batch)

    def classifier(self, graph_x):
        h = self.act(self.lin1(graph_x))
        return self.lin2(h)

    def forward(self, x, edge_index, batch):
        graph_x = self.get_graph_rep(x, edge_index, batch)
        return self.classifier(graph_x)

    def get_pred(self, x, edge_index, batch):
        graph_x = self.get_graph_rep(x, edge_index, batch)
        logits = self.classifier(graph_x)
        probs = self.softmax(logits)
        self.readout = probs
        return probs, logits

    def get_pred_explain(self, x, edge_index, edge_mask, batch, mask_is_logit=False):
        edge_weight = (edge_mask * EPS).sigmoid() if mask_is_logit else edge_mask
        graph_x = self.get_graph_rep(x, edge_index, batch, edge_weight=edge_weight)
        logits = self.classifier(graph_x)
        probs = self.softmax(logits)
        self.readout = probs
        return probs, logits


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_dataset = MUTAG188(args.data_path, mode="training")
    val_dataset = MUTAG188(args.data_path, mode="evaluation")
    test_dataset = MUTAG188(args.data_path, mode="testing")

    if args.random_label:
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for graph in dataset:
                graph.y.fill_(random.choice([0, 1]))

    model = Mutag188_GCN(
        in_channels=train_dataset.num_features,
        num_unit=args.num_unit,
        hidden_dim=128,
        num_classes=2,
        dropout=0.2,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4)
    min_error = None
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=CrossEntropyLoss())

        _, train_acc = Gtest(train_loader, model, device=device, criterion=CrossEntropyLoss())
        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=CrossEntropyLoss())
        test_error, test_acc = Gtest(test_loader, model, device=device, criterion=CrossEntropyLoss())
        scheduler.step(val_error)

        if min_error is None or val_error < min_error:
            min_error = val_error
            best_model_wts = copy.deepcopy(model.state_dict())

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader, model, device=device, criterion=CrossEntropyLoss())
            t3 = time.time()
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, "
                "Test acc: {:.5f}".format(epoch, t3 - t1, lr, loss, test_error, test_acc)
            )
            continue

        print(
            "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, "
            "Validation acc: {:5f}".format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc)
        )

    model.load_state_dict(best_model_wts)

    save_path = "mutag188_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
