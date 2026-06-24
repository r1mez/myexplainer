import argparse
import os
import os.path as osp
import random
import time
import sys
# 基于脚本位置加入项目根目录，避免依赖当前工作目录（直接 python /path/to/proteins_gnn.py 时也能找到 datasets）
_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, ModuleList, ReLU, Softmax, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from datasets.proteins_dataset import PROTEINS
from utils import Gtest, Gtrain, set_seed
import copy

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train PROTEINS Model")

    parser.add_argument(
        "--data_path",
        nargs="?",
        default=osp.join(osp.dirname(__file__), "..", "data", "proteins"),
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
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--verbose", type=int, default=10, help="Interval of evaluation.")
    parser.add_argument("--num_unit", type=int, default=3, help="number of Convolution layers(units)")
    parser.add_argument(
        "--random_label", type=bool, default=False, help="train a model under label randomization for sanity check"
    )
    return parser.parse_args()

from gnns.base import BaseGNNClassifier


class PROTEINSGCN(BaseGNNClassifier):
    """
    PROTEINS 图分类模型（GCN）：
      - 优化：引入 gnn.py 核心架构，将 Readout 改为 global_max_pool，精简分类头为单层 Linear。
      - 接口：完全保持原样，兼容后续解释器调用。
    """
    def __init__(
        self,
        in_channels: int,
        num_unit: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(in_channels, hidden_dim, num_classes)

        self.num_unit = num_unit
        self.dropout = dropout

        # ---- GCN 卷积层 (保持不变) ----
        self.convs = ModuleList()
        self.bns   = ModuleList()
        self.act   = ReLU()

        for i in range(num_unit):
            in_c = in_channels if i == 0 else hidden_dim
            out_c = hidden_dim
            conv = GCNConv(in_c, out_c)
            bn   = BatchNorm1d(out_c)
            self.convs.append(conv)
            self.bns.append(bn)

        # ---- 图级分类头 (优化：对齐 gnn.py 使用单层 Linear) ----
        self.fc = Linear(hidden_dim, num_classes)
        self.softmax = Softmax(dim=1)

        self.reset_parameters()

    # ----------------- 基础模块 -----------------
    def reset_parameters(self):
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()
        self.fc.reset_parameters()  # 仅需重置单层 FC

    def _forward_convs(self, x, edge_index, edge_weight=None):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), device=edge_index.device, dtype=x.dtype
            )
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_weight)
            h = bn(h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _pool(self, node_emb, batch):
        return global_max_pool(node_emb, batch)

    def _classify(self, graph_emb):
        return self.fc(graph_emb)


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_dataset = PROTEINS(args.data_path, mode="training")
    val_dataset = PROTEINS(args.data_path, mode="evaluation")
    test_dataset = PROTEINS(args.data_path, mode="testing")

    in_channels = train_dataset.num_features

    model = PROTEINSGCN(
        in_channels=in_channels,
        num_unit=args.num_unit,
        hidden_dim=64,
        num_classes=2,
        dropout=0.4,
    ).to(device)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4)
    min_error = None
    best_model_wts = copy.deepcopy(model.state_dict()) # 初始化最佳模型权重

    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=CrossEntropyLoss())

        _, train_acc = Gtest(train_loader, model, device=device, criterion=CrossEntropyLoss())

        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=CrossEntropyLoss())
        test_error, test_acc = Gtest(test_loader, model, device=device, criterion=CrossEntropyLoss())
        scheduler.step(val_error)
        
        # 判断并保存验证集上的最佳模型权重
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

    save_path = "proteins_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    # 此时保存的是最佳模型
    torch.save(model.cpu(), osp.join(args.model_path, save_path))