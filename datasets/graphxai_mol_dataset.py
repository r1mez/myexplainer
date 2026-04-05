import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import numpy as np
import torch
from numpy.random import RandomState
from torch_geometric.data import InMemoryDataset

from .ben_dataset import read_data


class GraphXAIMoleculeDataset(InMemoryDataset):
    """Shared GraphXAI-style NPZ dataset wrapper with fixed train/val/test splits."""

    dataset_name = None
    down_sample = False
    splits = ["training", "evaluation", "testing"]

    def __init__(
        self,
        root: str,
        mode: str = "testing",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        if self.dataset_name is None:
            raise ValueError("dataset_name must be defined on GraphXAIMoleculeDataset subclasses.")
        assert mode in self.splits
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index(f"{mode}.pt")
        processed_path = self.processed_paths[idx]
        if not osp.exists(processed_path):
            # Older PyG setups or partially-created processed folders can leave the
            # requested split missing even after Dataset.__init__. Re-run the normal
            # download/process hooks once so the dataset can self-heal.
            self._download()
            self._process()

        if not osp.exists(processed_path):
            available_files = []
            if osp.isdir(self.processed_dir):
                available_files = sorted(os.listdir(self.processed_dir))
            raise FileNotFoundError(
                f"Processed split '{processed_path}' was not created for dataset "
                f"'{self.dataset_name}'. Existing processed files: {available_files}"
            )

        self.data, self.slices, self.sizes = torch.load(processed_path)

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{self.dataset_name}.npz"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self) -> None:
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if osp.exists(raw_path):
            print(f"Using existing raw data: {raw_path}")
            return

        project_root = osp.dirname(osp.dirname(__file__))
        candidate_paths = [
            osp.join(project_root, "tmp_files", self.dataset_name, f"{self.dataset_name}.npz"),
            osp.join(self.root, f"{self.dataset_name}.npz"),
            osp.join(project_root, "data", self.dataset_name, f"{self.dataset_name}.npz"),
        ]
        source_path = next((path for path in candidate_paths if osp.exists(path)), None)
        if source_path is None:
            raise FileNotFoundError(
                f"{self.dataset_name}.npz was not found. Checked raw target '{raw_path}' "
                f"and source candidates: {candidate_paths}"
            )

        os.makedirs(self.raw_dir, exist_ok=True)
        shutil.copy2(source_path, raw_path)
        print(f"Copied raw data to: {raw_path}")

    def process(self) -> None:
        data_list, _, sizes = read_data(
            self.dataset_name,
            self.raw_dir,
            down_sample=self.down_sample,
        )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        num_graphs = len(data_list)
        indices = np.arange(num_graphs)
        prng = RandomState(42)
        prng.shuffle(indices)

        shuffled_data = []
        for graph_idx, orig_idx in enumerate(indices):
            data = data_list[int(orig_idx)]
            data.idx = graph_idx
            data.name = f"{self.dataset_name}_{graph_idx}"
            shuffled_data.append(data)

        train_end = int(0.8 * num_graphs)
        eval_end = int(0.9 * num_graphs)
        split_to_data = {
            "training": shuffled_data[:train_end],
            "evaluation": shuffled_data[train_end:eval_end],
            "testing": shuffled_data[eval_end:],
        }

        for split_idx, split_name in enumerate(self.splits):
            split_data = split_to_data[split_name]
            data, slices = self.collate(split_data)
            torch.save((data, slices, sizes), self.processed_paths[split_idx])
