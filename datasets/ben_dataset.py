import sys
import os.path as osp
import random
import sys
from typing import *

import numpy as np
import torch
from numpy.random import RandomState
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import coalesce

sys.path.append(osp.split(sys.path[0])[0])

def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    r"""
    Convert edge_mask to node_mask

    Args:
        node_mask (torch.Tensor): Boolean mask over all nodes included in edge_index. Indices must
            match to those in edge index. This is straightforward for graph-level prediction, but
            converting over subgraph must be done carefully to match indices in both edge_index and
            the node_mask.
        edge_index (torch.Tensor): adj of graph.
    """

    node_numbers = node_mask.nonzero(as_tuple=True)[0]

    iter_mask = torch.zeros((edge_index.shape[1],))

    # See if edges have both ends in the node mask
    for i in range(edge_index.shape[1]):
        iter_mask[i] = (edge_index[0, i] in node_numbers) and (edge_index[1, i] in node_numbers)

    return iter_mask.long()

def read_file(data_name: str, dir_path: str):
    r"""
    Code from https://github.com/mims-harvard/GraphXAI, I just rearrange this.
    Returns:
        all_graphs (list of `torch_geometric.data.Data`): List of all graphs in the
            dataset
        explanations (list of `Explanation`): List of all ground-truth explanations for
            each corresponding graph. Ground-truth explanations consist of multiple
            possible explanations as some of the molecular prediction tasks consist
            of multiple possible pathways to predicting a given label.
        zinc_ids (list of ints): Integers that map each molecule back to its original
            ID in the ZINC dataset.
    """
    data_path = osp.join(dir_path, f'{data_name}.npz')
    data = np.load(data_path, allow_pickle=True)

    att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
    y_list = [y[i][0] for i in range(y.shape[0])]

    X = X[0]

    # Unique zinc identifiers:
    zinc_ids = df[:, 1]

    all_graphs = []
    explanations = []

    for i in range(len(X)):
        x = torch.from_numpy(X[i]['nodes'])
        edge_attr = torch.from_numpy(X[i]['edges'])
        y = torch.tensor([y_list[i]], dtype=torch.long)
        # Get edge_index:
        e1 = torch.from_numpy(X[i]['receivers']).long()
        e2 = torch.from_numpy(X[i]['senders']).long()

        edge_index = torch.stack([e1, e2])

        data_i = Data(
            x=x,
            y=y,
            edge_attr=edge_attr,
            edge_index=edge_index
        )

        all_graphs.append(data_i)  # Add to larger list

        # Get ground-truth explanation:
        node_imp = torch.from_numpy(att[i][0]['nodes']).float()

        # Error-check:
        assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
        assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0],
                                                                                    x.shape[0]) \
                                                + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])

        i_exps = []
        node_imp_ = torch.zeros(data_i.num_nodes).long()
        edge_imp_ = torch.zeros(data_i.num_edges).long()
        for j in range(node_imp.shape[1]):
            # if there have different ground truth, put all of them
            node_imp_ = torch.bitwise_or(node_imp_, node_imp[:, j].long())
            edge_imp_ = torch.bitwise_or(edge_imp_,
                                         edge_mask_from_node_mask(node_imp[:, j].bool(), edge_index=edge_index))
        # i_exps.append(node_imp_, edge_imp_])
        explanations.append([node_imp_, edge_imp_])

    return all_graphs, explanations, zinc_ids

def read_data(name: str, dir_path: str, down_sample: bool):
    data_list, exp_list, zinc_idx = read_file(name, dir_path)
    if down_sample:
        # down_samples because of extreme imbalance
        zero_bin = [i for i, data in enumerate(data_list) if data.y == 0]
        one_bin  = [i for i, data in enumerate(data_list) if data.y == 1]
        random.seed(2024)
        keep_idxs = random.sample(zero_bin, k = 2 * len(one_bin)) + one_bin
        data_list = [data_list[i] for i in keep_idxs]
        exp_list  = [exp_list[i] for i in keep_idxs]

    node_slice, edge_slice = [0], [0]
    for data, [node_imp, edge_imp] in zip(data_list, exp_list):
        edge_attrs = [data.edge_attr, edge_imp]
        edge_index, edge_attrs = coalesce(data.edge_index, edge_attrs, data.num_nodes)
        data.edge_index = edge_index
        data.edge_attr = edge_attrs[0]
        data.edge_mask = edge_attrs[1]
        data.node_mask = node_imp
        node_slice.append(data.num_nodes)
        edge_slice.append(data.num_edges)
    node_slice = torch.cumsum(torch.tensor(node_slice, dtype=torch.long), dim=0)
    edge_slice = torch.cumsum(torch.tensor(edge_slice, dtype=torch.long), dim=0)
    slices = {
        'edge_index': edge_slice,
        'x': node_slice,
        'edge_attr': edge_slice,
        'edge_mask': edge_slice,
        'node_mask': node_slice,
        'y': torch.arange(0, len(data_list) + 1, dtype=torch.long)
    }
    sizes = {
        'num_node_labels': data_list[0].x.size(-1),
        'num_edge_labels': data_list[0].edge_attr.size(-1),
        'num_classes': 2
    }
    return data_list, slices, sizes

class BaseDataset(InMemoryDataset):
    """
    A base class to process datasets which we need in GNN explaining.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """
        Which file the processed datasets should be saved.
        """
        return ['data.pt']

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class Benzene(BaseDataset):
    # 必须在类级别定义 name，或者在 __init__ 中定义
    name = 'benzene'

    def __init__(
            self,
            root_path: str,
            mode: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
    ):
        self.mode = mode
        # ✅ 直接使用 root_path，不要重复添加 "benzene" 子目录
        # 因为在 utils/dataset.py 的 get_datasets() 中已经添加过了
        root = root_path
        super().__init__(root, transform, pre_transform, pre_filter)
        # 加载对应的 pt 文件
        # 注意：这里加载的是 process 中保存的格式
        self.data, self.slices, self.sizes = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        # 这样 get_datasets 调用不同的 mode 时，PyG 会识别到不同的 pt 文件
        return [f'{self.mode}_data.pt']

    def process(self):
        # 1. 直接使用类属性 self.name，确保它已定义
        # raw_dir 默认是 root/raw
        data_list, _, sizes = read_data(self.name, self.raw_dir, down_sample=False)

        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            max_num_nodes = max(data.num_nodes for data in data_list)
            data_list = [self.pre_transform(d, max_num_nodes) for d in data_list]

        # 2. 关键：固定随机种子确保 train/val/test 划分一致
        num_graphs = len(data_list)
        indices = np.arange(num_graphs)
        prng = RandomState(42)
        prng.shuffle(indices)
        data_list = [data_list[i] for i in indices]

        # 3. 划分
        train_end = int(0.8 * num_graphs)
        val_end = int(0.9 * num_graphs)

        if self.mode == "training":
            final_list = data_list[:train_end]
        elif self.mode == "testing":
            final_list = data_list[train_end:val_end]
        elif self.mode == "evaluation":
            final_list = data_list[val_end:]
        else:
            final_list = data_list

        # 4. 转换并保存
        data, slices = self.collate(final_list)
        print(f"Saving {self.mode} dataset to {self.processed_paths[0]}")
        torch.save((data, slices, sizes), self.processed_paths[0])





if __name__ == '__main__':
    path = "/home/ll_yqs2/data/Projects/myexplainer/data/benzene"
    print("path: ", path)
    dataset = Benzene(path, mode="training")
    dataset = Benzene(path, mode="testing")
    dataset = Benzene(path, mode="evaluation")
    print(dataset)