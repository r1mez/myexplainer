import os
import os.path as osp
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip


class MUTAG188(InMemoryDataset):
    """Classic 188-graph MUTAG dataset in TU Dortmund text format."""

    url = "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip"
    splits = ["training", "evaluation", "testing"]

    _RAW_BASE_NAMES = [
        "MUTAG_A.txt",
        "MUTAG_edge_labels.txt",
        "MUTAG_graph_indicator.txt",
        "MUTAG_graph_labels.txt",
        "MUTAG_node_labels.txt",
    ]

    def __init__(
        self, root, mode="testing", transform=None, pre_transform=None, pre_filter=None
    ):
        assert mode in self.splits
        self.mode = mode
        self._ensure_raw_data_for_root(root)
        super(MUTAG188, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        try:
            self.data, self.slices = torch.load(
                self.processed_paths[idx], weights_only=False
            )
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        prefix = self._find_raw_prefix(self.raw_dir)
        if prefix:
            return [osp.join(prefix, name) for name in self._RAW_BASE_NAMES]
        return list(self._RAW_BASE_NAMES)

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    @staticmethod
    def _project_root():
        return osp.abspath(osp.join(osp.dirname(__file__), os.pardir))

    @classmethod
    def _bundled_source_dir(cls):
        return osp.join(cls._project_root(), "tmp_files", "mutag188")

    @classmethod
    def _ensure_raw_data_for_root(cls, root):
        raw_dir = osp.join(osp.expanduser(str(root)), "raw")
        if cls._has_raw_data(raw_dir):
            return

        source_dir = cls._bundled_source_dir()
        if not all(osp.isfile(osp.join(source_dir, name)) for name in cls._RAW_BASE_NAMES):
            return

        os.makedirs(raw_dir, exist_ok=True)
        for name in cls._RAW_BASE_NAMES + ["README.txt"]:
            source = osp.join(source_dir, name)
            if osp.isfile(source):
                shutil.copyfile(source, osp.join(raw_dir, name))

    @classmethod
    def _has_raw_data(cls, raw_dir):
        for subdir in ("", "MUTAG", "mutag", "mutag188"):
            if all(osp.isfile(osp.join(raw_dir, subdir, name)) for name in cls._RAW_BASE_NAMES):
                return True
        return False

    @classmethod
    def _find_raw_prefix(cls, raw_dir):
        for subdir in ("", "MUTAG", "mutag", "mutag188"):
            if all(osp.isfile(osp.join(raw_dir, subdir, name)) for name in cls._RAW_BASE_NAMES):
                return subdir
        return ""

    @staticmethod
    def _as_1d(array):
        array = np.asarray(array)
        if array.ndim == 0:
            return np.array([array.item()])
        return array.reshape(-1)

    @classmethod
    def _encode_labels(cls, labels):
        labels = cls._as_1d(labels).astype(int)
        unique = sorted(np.unique(labels).tolist())
        label_to_idx = {label: idx for idx, label in enumerate(unique)}
        encoded = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
        return torch.from_numpy(encoded).long()

    @classmethod
    def _encode_graph_labels(cls, labels):
        """Align MUTAG188 class ids with the 4337-graph Mutagenicity dataset.

        Project-wide target semantics:
          0 -> mutagen (positive)
          1 -> nonmutagen (negative)

        The TU Dortmund MUTAG raw labels are typically encoded as:
          1  -> mutagen
          -1 -> nonmutagen
        """

        labels = cls._as_1d(labels).astype(int)
        unique = sorted(np.unique(labels).tolist())

        if unique == [-1, 1]:
            label_to_idx = {
                1: 0,
                -1: 1,
            }
        elif unique == [0, 1]:
            label_to_idx = {
                0: 0,
                1: 1,
            }
        else:
            raise ValueError(
                "Unsupported MUTAG188 graph label set "
                f"{unique}. Expected either [-1, 1] or [0, 1]."
            )

        encoded = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
        return torch.from_numpy(encoded).long()

    @classmethod
    def _one_hot_labels(cls, labels):
        labels = cls._as_1d(labels).astype(int)
        if labels.size == 0:
            return torch.empty((0, 0), dtype=torch.float)

        if labels.min() >= 0:
            encoded = torch.from_numpy(labels).long()
            num_classes = int(labels.max()) + 1
        else:
            encoded = cls._encode_labels(labels)
            num_classes = int(encoded.max().item()) + 1

        return F.one_hot(encoded, num_classes=num_classes).float()

    def download(self):
        if self._has_raw_data(self.raw_dir):
            print("Using existing mutag188 raw data:", self.raw_dir)
            return

        self._ensure_raw_data_for_root(self.root)
        if self._has_raw_data(self.raw_dir):
            print("Copied mutag188 raw data from:", self._bundled_source_dir())
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        raw_paths = [osp.join(self.raw_dir, name) for name in self.raw_file_names]

        edge_array = np.loadtxt(raw_paths[0], delimiter=",", dtype=np.int64)
        if edge_array.ndim == 1:
            edge_array = edge_array.reshape(1, -1)
        edge_array = edge_array - 1

        edge_labels = np.loadtxt(raw_paths[1], dtype=np.int64)
        edge_attr_all = self._one_hot_labels(edge_labels)

        graph_indicator = self._as_1d(np.loadtxt(raw_paths[2], dtype=np.int64))
        graph_labels = np.loadtxt(raw_paths[3], dtype=np.int64)
        y = self._encode_graph_labels(graph_labels).view(-1, 1)

        node_labels = self._as_1d(np.loadtxt(raw_paths[4], dtype=np.int64))
        x_all = self._one_hot_labels(node_labels)

        num_graphs = int(y.size(0))
        data_list = []

        for graph_idx in range(num_graphs):
            graph_id = graph_idx + 1
            node_indices = np.where(graph_indicator == graph_id)[0]
            if node_indices.size == 0:
                continue

            edge_mask = (
                (graph_indicator[edge_array[:, 0]] == graph_id)
                & (graph_indicator[edge_array[:, 1]] == graph_id)
            )
            local_edges = edge_array[edge_mask] - int(node_indices.min())
            if local_edges.size == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, edge_attr_all.size(1)), dtype=torch.float)
            else:
                edge_index = torch.from_numpy(local_edges.T).long().contiguous()
                edge_attr = edge_attr_all[torch.from_numpy(edge_mask)]

            data = Data(
                x=x_all[torch.from_numpy(node_indices).long()],
                y=y[graph_idx],
                z=torch.from_numpy(node_labels[node_indices]).long(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                name="mutag188_%d" % graph_idx,
                idx=graph_idx,
                num_nodes=int(node_indices.size),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        rng = random.Random(0)
        rng.shuffle(data_list)

        num_total = len(data_list)
        num_test = max(1, round(num_total * 0.1)) if num_total >= 3 else 0
        num_val = max(1, round(num_total * 0.1)) if num_total >= 3 else 0
        if num_test + num_val >= num_total:
            num_test = max(0, min(num_test, num_total - 1))
            num_val = max(0, min(num_val, num_total - num_test - 1))

        test_data = data_list[:num_test]
        val_data = data_list[num_test:num_test + num_val]
        train_data = data_list[num_test + num_val:]

        print(
            "Total mutag188 graphs: "
            f"{num_total} (train={len(train_data)}, "
            f"evaluation={len(val_data)}, testing={len(test_data)})"
        )

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
