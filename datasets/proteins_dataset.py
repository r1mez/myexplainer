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

    def __init__(
        self, root, mode="testing", transform=None, pre_transform=None, pre_filter=None
    ):
        assert mode in self.splits
        self.mode = mode
        super(PROTEINS, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return [
            "PROTEINS/" + i
            for i in [
                "PROTEINS_A.txt",              # 边列表
                "PROTEINS_graph_indicator.txt",# 图-节点对应
                "PROTEINS_graph_labels.txt",   # 图标签
                "PROTEINS_node_labels.txt",    # 节点标签（可选）
                "PROTEINS_node_attributes.txt" # 节点属性（连续特征，PROTEINS 有此文件）
            ]
        ]

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if os.path.exists(osp.join(self.raw_dir, "PROTEINS")):
            print("Using existing data in folder PROTEINS")
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

        # 读取节点属性（PROTEINS 使用 node_attributes 而非离散 node_labels）
        # 如果文件不存在或想只用 one-hot 节点标签，可切换为 node_labels
        try:
            node_attr = np.loadtxt(
                osp.join(self.raw_dir, self.raw_file_names[4]), delimiter=","
            )
            x = torch.tensor(node_attr, dtype=torch.float)
        except:
            #  fallback: 使用 node_labels one-hot（如果存在）
            node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]))
            encoder = preprocessing.OneHotEncoder().fit(
                np.unique(node_label).reshape(-1, 1)
            )
            x = encoder.transform(node_label.reshape(-1, 1)).toarray()
            x = torch.tensor(x, dtype=torch.float)

        # 图指示器（哪个节点属于哪个图）
        z = np.loadtxt(
            osp.join(self.raw_dir, self.raw_file_names[1]), dtype=int
        )

        # 图标签（二分类：酶 / 非酶）
        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]))
        y = torch.unsqueeze(torch.LongTensor(y), 1).long()

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
                x=x[perm],                    # 节点特征（连续属性或 one-hot）
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
        torch.save(self.collate(data_list[400:]), self.processed_paths[0])   # training (剩余部分)
        torch.save(self.collate(data_list[200:400]), self.processed_paths[1]) # evaluation
        torch.save(self.collate(data_list[:200]), self.processed_paths[2])     # testing