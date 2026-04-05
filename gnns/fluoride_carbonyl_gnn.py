import argparse
import copy
import os
import os.path as osp
import sys
import time

sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from datasets.fluoride_carbonyl_dataset import FluorideCarbonyl
from utils import Gtest, Gtrain, set_seed

try:
    from .alkane_carbonyl_gnn import AlkaneCarbonylGCN, get_class_weights
except ImportError:
    from alkane_carbonyl_gnn import AlkaneCarbonylGCN, get_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fluoride-Carbonyl Model")

    parser.add_argument(
        "--data_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "data", "fluoride_carbonyl"),
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
    return parser.parse_args()


class FluorideCarbonylGCN(AlkaneCarbonylGCN):
    pass


if __name__ == "__main__":
    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    test_dataset = FluorideCarbonyl(args.data_path, mode="testing")
    val_dataset = FluorideCarbonyl(args.data_path, mode="evaluation")
    train_dataset = FluorideCarbonyl(args.data_path, mode="training")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = FluorideCarbonylGCN(
        in_channels=train_dataset[0].x.size(1),
        hidden_dim=args.hidden_dim,
        conv_unit=args.num_unit,
        dropout=args.dropout,
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

    save_path = "fluoride_carbonyl_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
