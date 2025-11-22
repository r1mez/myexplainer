import argparse
import os
import os.path as osp
import random
import time
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, ModuleList, ReLU, Softmax, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from datasets import BA2Motif
from utils import Gtest, Gtrain, set_seed
from torch_geometric.utils import degree

EPS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Train BA-2Motif Model")

    parser.add_argument(
        "--data_path", nargs="?", default=osp.join(osp.dirname(__file__), "..", "data", "ba2motif"), help="Input data path."
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


class BA2MotifGCN(torch.nn.Module):
    """
    只用基础 GCN 的 BA-2Motif 模型：

      - 输入：节点特征 x（in_channels 维）
      - 若数据集本身没有特征，可预先给 x 设置为常量/one-hot 等
      - 网络结构：多层 GCNConv + BN + ReLU + Dropout + global_mean_pool + MLP 分类
      - get_pred_explain 接口用于解释器：通过 edge_weight 控制边的重要性
    """
    def __init__(
        self,
        in_channels: int,
        num_unit: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.num_unit = num_unit
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # ---- GCN 卷积层 ----
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

        # ---- 图级分类头 ----
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

        self.reset_parameters()

    # ----------------- 基础模块 -----------------
    def reset_parameters(self):
        for conv, bn in zip(self.convs, self.bns):
            conv.reset_parameters()
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_node_reps(self, x, edge_index, edge_weight=None):
        """
        得到节点表示：
          - x: [N, in_channels]
          - edge_index: [2, E]
          - edge_weight: [E]，可为 None / 全 1 / 掩码权重
        """
        h = x
        # 逐层 GCN + BN + ReLU + Dropout
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = bn(h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [N, hidden_dim]

    def get_graph_rep(self, x, edge_index, batch, edge_weight=None):
        """
        图级表示：节点表示做 global_mean_pool。
        """
        node_x = self.get_node_reps(x, edge_index, edge_weight=edge_weight)
        graph_x = global_mean_pool(node_x, batch)  # [B, hidden_dim]
        return graph_x

    def classifier(self, graph_x):
        """
        分类头：graph 表示 -> logits
        """
        h = self.relu(self.lin1(graph_x))
        logits = self.lin2(h)  # [B, num_classes]
        return logits

    # ----------------- 对外接口 -----------------
    def forward(self, x, edge_index, batch):
        """
        标准训练/推理接口：返回 logits（给 CrossEntropyLoss 用）
        """
        graph_x = self.get_graph_rep(x, edge_index, batch, edge_weight=None)
        logits = self.classifier(graph_x)
        return logits

    def get_pred(self, x, edge_index, batch):
        """
        推理用：返回 (softmax 概率, logits)
        """
        graph_x = self.get_graph_rep(x, edge_index, batch, edge_weight=None)
        logits = self.classifier(graph_x)
        probs  = self.softmax(logits)
        self.readout = probs
        return probs, logits

    def get_pred_explain(self, x, edge_index, edge_mask, batch, mask_is_logit=False):
        """
        解释用接口：
          - edge_mask:
              1) 若为 logits（如 PROXYExplainer 输出），则 mask_is_logit=True，内部走 sigmoid
              2) 若已在 [0,1]（如 MyExplainerV2 的 adj_recon[row, col]），则 mask_is_logit=False

        返回: (softmax 概率, logits)
        """
        if mask_is_logit:
            edge_weight = (edge_mask * EPS).sigmoid()
        else:
            edge_weight = edge_mask

        graph_x = self.get_graph_rep(x, edge_index, batch, edge_weight=edge_weight)
        logits  = self.classifier(graph_x)
        probs   = self.softmax(logits)
        self.readout = probs
        return probs, logits

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_dataset = BA2Motif(args.data_path, mode="training")
    val_dataset = BA2Motif(args.data_path, mode="evaluation")
    test_dataset = BA2Motif(args.data_path, mode="testing")

    in_channels = train_dataset.num_features  # PyG 常规接口

    model = BA2MotifGCN(
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
    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=CrossEntropyLoss())

        _, train_acc = Gtest(train_loader, model, device=device, criterion=CrossEntropyLoss())

        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=CrossEntropyLoss())
        test_error, test_acc = Gtest(test_loader, model, device=device, criterion=CrossEntropyLoss())
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

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
    save_path = "ba2motif_gcn.pt"
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
