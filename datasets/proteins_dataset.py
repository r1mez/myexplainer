import os
import os.path as osp
import random

import numpy as np
import sklearn.preprocessing as preprocessing
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip


class PROTEINS(InMemoryDataset):
    url = "https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip"

    splits = ["training", "evaluation", "testing"]

    # 相对 raw 目录的文件名（可与官方 zip 解压布局对齐）
    _RAW_BASE_NAMES = [
        "PROTEINS_A.txt",  # 边列表
        "PROTEINS_graph_indicator.txt",
        "PROTEINS_graph_labels.txt",
        "PROTEINS_node_labels.txt",
        "PROTEINS_node_attributes.txt",
    ]

    @staticmethod
    def _has_raw_data(raw_dir):
        """是否已有原始数据（支持 raw 根目录平铺、PROTEINS/、proteins/ 三种常见布局）。"""
        fn = "PROTEINS_A.txt"
        if osp.isfile(osp.join(raw_dir, fn)):
            return True
        for sub in ("PROTEINS", "proteins"):
            if osp.isfile(osp.join(raw_dir, sub, fn)):
                return True
        return False

    @staticmethod
    def _find_raw_prefix(raw_dir):
        """返回相对 raw 的路径前缀：'' 表示文件直接在 raw 下；否则为子目录名。无数据时假定官方 zip 为 PROTEINS/。"""
        fn = "PROTEINS_A.txt"
        if osp.isfile(osp.join(raw_dir, fn)):
            return ""
        for sub in ("PROTEINS", "proteins"):
            if osp.isfile(osp.join(raw_dir, sub, fn)):
                return sub
        return "PROTEINS"

    def __init__(
        self, root, mode="testing", transform=None, pre_transform=None, pre_filter=None
    ):
        assert mode in self.splits
        self.mode = mode
        raw_dir = osp.join(osp.expanduser(root), "raw")
        self._raw_prefix = self._find_raw_prefix(raw_dir)
        super(PROTEINS, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        p = self._raw_prefix
        if p:
            return [osp.join(p, n) for n in self._RAW_BASE_NAMES]
        return list(self._RAW_BASE_NAMES)

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if self._has_raw_data(self.raw_dir):
            print("使用已有原始数据，跳过下载：", self.raw_dir)
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        # 读取边
        edge_index = np.loadtxt(
            osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=","
        ).T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)  # node idx from 0

        # 节点特征：使用离散 node_labels 的 one-hot，与 Mutagenicity / NCI1 等一致。
        # （连续 node_attributes 会导致 x 维度过小或语义非类别，GenGraphEx 的 GraphRepModelDiscrete 需要 one-hot + argmax。）
        path_labels = osp.join(self.raw_dir, self.raw_file_names[3])
        node_label = np.loadtxt(path_labels)
        if node_label.ndim == 0:
            node_label = np.array([node_label.item()])
        elif node_label.ndim > 1:
            node_label = node_label.ravel()
        encoder = preprocessing.OneHotEncoder().fit(
            np.unique(node_label).reshape(-1, 1)
        )
        x = encoder.transform(node_label.reshape(-1, 1)).toarray()
        x = torch.tensor(x, dtype=torch.float)

        # 图指示器（哪个节点属于哪个图）
        z = np.loadtxt(
            osp.join(self.raw_dir, self.raw_file_names[1]), dtype=int
        )

        # 图标签（二分类：官方 TUDataset 为 1/2，需转为 0/1 以配合 CrossEntropyLoss）
        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]))
        y = torch.tensor(y, dtype=torch.long).view(-1, 1) - 1

        num_graphs = len(y)
        total_edges = edge_index.size(1)
        begin = 0

        data_list = []

        for i in range(num_graphs):
            # 当前图的节点索引
            perm = np.where(z == i + 1)[0]
            if len(perm) == 0:
                continue

            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            # 当前图的边
            graph_edge_index = edge_index[:, begin:end] - int(min(perm))

            data = Data(
                x=x[perm],                    # 节点特征（node_labels 的 one-hot）
                y=y[i],                       # 图标签
                edge_index=graph_edge_index,
                idx=i,                        # 可选：图编号
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                begin = end
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            begin = end
            data_list.append(data)

        print(f"Total graphs processed: {len(data_list)}")  # 应为 1113

        # 随机打乱
        random.shuffle(data_list)

        # 按照你提供的其他数据集风格划分（前 500 测试，中间 500 验证，后面的训练）
        # PROTEINS 总共 1113 个图，可根据需要调整比例
        torch.save(self.collate(data_list[224:]), self.processed_paths[0])   # training (剩余部分)
        torch.save(self.collate(data_list[112:224]), self.processed_paths[1]) # evaluation
        torch.save(self.collate(data_list[:112]), self.processed_paths[2])     # testing