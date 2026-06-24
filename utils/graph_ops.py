"""Pure graph-theoretic operations: subgraph extraction from original vs counterfactual."""

from typing import List, Union

import torch
from torch_geometric.data import Data, Batch


def _normalize_input(inp: Union[Data, Batch, List[Data]]) -> List[Data]:
    """Normalize input to a list of Data objects."""
    if isinstance(inp, Data):
        return [inp]
    elif isinstance(inp, Batch):
        return inp.to_data_list()
    elif isinstance(inp, list):
        if not all(isinstance(g, Data) for g in inp):
            raise ValueError("List must contain Data objects.")
        return inp
    else:
        raise ValueError("Input must be Data, Batch, or list of Data.")


def extract_explanatory_subgraph(
    original: Union[Data, Batch, List[Data]],
    counterfactual: Union[Data, Batch, List[Data]]
) -> Union[Data, Batch, List[Data]]:
    """
    解释子图：只基于边集变化（无向意义上的边集差异）。

    规则：
    - 解释边 =
        * 原图有但反事实没有的边
        * 反事实有但原图没有的边
      （用无向 canonical 形式比较：{min(u,v), max(u,v)}）
    - 解释子图的节点特征来自原图。
    - 如果原图和反事实图在 canonical 边集上一样：
        -> 返回的解释子图保留原图所有节点（x不变），但 edge_index 为空。
    - 有变化时：
        -> 只保留"涉及变化边"的端点节点，然后重新编号为紧凑的 0..N-1。
    """

    def _process_pair(orig: Data, cf: Data) -> Data:
        if orig.num_nodes != cf.num_nodes:
            raise ValueError(f"Graph pair mismatch: {orig.num_nodes} vs {cf.num_nodes} nodes.")

        device = orig.x.device
        num_nodes, feat_dim = orig.x.shape

        # 无向 canonical 边集：用于判断边是否存在（忽略方向/多重边）
        def canonical_edges(edge_idx: torch.Tensor) -> set:
            if edge_idx.numel() == 0:
                return set()
            src, tgt = edge_idx
            mins = torch.min(src, tgt)
            maxs = torch.max(src, tgt)
            return {(int(mn), int(mx)) for mn, mx in zip(mins, maxs)}

        orig_canon = canonical_edges(orig.edge_index)
        cf_canon = canonical_edges(cf.edge_index)

        # 如果无向边集完全相同：解释子图 = 所有原图节点 + 空边集
        if orig_canon == cf_canon:
            empty_eidx = torch.empty((2, 0), dtype=torch.long, device=device)
            # 保留原图节点特征
            return Data(x=orig.x.clone(), edge_index=empty_eidx)

        explain_edges = []

        # 1) 原图有、反事实没有的边（保留原图方向）
        src_o, tgt_o = orig.edge_index
        for i in range(orig.edge_index.size(1)):
            u = int(src_o[i])
            v = int(tgt_o[i])
            key = (min(u, v), max(u, v))
            if key not in cf_canon:
                explain_edges.append(orig.edge_index[:, i:i + 1])

        # 2) 反事实有、原图没有的边（保留反事实方向）
        src_c, tgt_c = cf.edge_index
        for i in range(cf.edge_index.size(1)):
            u = int(src_c[i])
            v = int(tgt_c[i])
            key = (min(u, v), max(u, v))
            if key not in orig_canon:
                # 注意把边搬到 orig 所在 device
                explain_edges.append(cf.edge_index[:, i:i + 1].to(device=device))

        # 理论上 orig_canon != cf_canon 时 explain_edges 一定非空，但做个兜底
        if explain_edges:
            explain_eidx = torch.cat(explain_edges, dim=1)
        else:
            explain_eidx = torch.empty((2, 0), dtype=torch.long, device=device)

        # 从解释边的端点收集需要保留的节点
        nodes_set = set()
        if explain_eidx.numel() > 0:
            nodes_set.update(int(u) for u in explain_eidx[0])
            nodes_set.update(int(v) for v in explain_eidx[1])

        # 极端兜底：如果 nodes_set 竟然为空，就退化成"保留所有节点、无边"
        if not nodes_set:
            empty_eidx = torch.empty((2, 0), dtype=torch.long, device=device)
            return Data(x=orig.x.clone(), edge_index=empty_eidx)

        node_list = sorted(nodes_set)
        node_map = {old: new for new, old in enumerate(node_list)}

        # 重新取节点特征（来自原图）
        idx_tensor = torch.tensor(node_list, dtype=torch.long, device=device)
        explain_x = orig.x[idx_tensor]

        # 重新编号解释边
        new_src = torch.tensor(
            [node_map[int(u)] for u in explain_eidx[0]],
            dtype=torch.long,
            device=device,
        )
        new_tgt = torch.tensor(
            [node_map[int(v)] for v in explain_eidx[1]],
            dtype=torch.long,
            device=device,
        )
        new_eidx = torch.stack([new_src, new_tgt], dim=0)

        return Data(x=explain_x, edge_index=new_eidx)

    orig_list = _normalize_input(original)
    cf_list = _normalize_input(counterfactual)

    if len(orig_list) != len(cf_list):
        raise ValueError("Original and counterfactual inputs must have matching number of graphs.")

    explain_list = [_process_pair(o, c) for o, c in zip(orig_list, cf_list)]

    # 返回类型与输入保持一致
    if len(explain_list) == 1:
        return explain_list[0]
    elif isinstance(original, Batch) or isinstance(counterfactual, Batch):
        return Batch.from_data_list(explain_list)
    else:
        return explain_list


def exclude_explanatory_subgraph(original: Union[Data, Batch, List[Data]],
                                 counterfactual: Union[Data, Batch, List[Data]]) -> Union[Data, Batch, List[Data]]:
    """
    Efficient batch-enabled non-explanatory subgraph extraction (excludes explanatory parts).

    Supports single Data, lists of Data, or Batch objects for both inputs.
    Processes each graph pair independently using vectorized operations where possible.
    For Batch inputs, uses to_data_list() for per-graph processing, then reconstructs Batch.
    Time complexity: O(sum(V + E) over all graphs), suitable for batched inference.

    Logic (vectorized where feasible):
    - Changed nodes: Vectorized mask via (original.x != counterfactual.x).any(-1).
    - Retained edges: From original, only if exists in counterfactual AND not both endpoints changed.
    - Node features: Original, but zeroed for changed nodes.
    - All nodes preserved; only edges filtered.
    """

    def _process_pair(orig: Data, cf: Data) -> Data:
        if orig.num_nodes != cf.num_nodes:
            raise ValueError(f"Graph pair mismatch: {orig.num_nodes} vs {cf.num_nodes} nodes.")

        device = orig.x.device
        num_nodes, feat_dim = orig.x.shape

        # Vectorized changed nodes mask
        changed_mask = ~(orig.x == cf.x).all(-1)  # bool tensor [num_nodes]
        changed_nodes = torch.nonzero(changed_mask).flatten()  # tensor [num_changed]
        changed_set = set(changed_nodes.tolist())

        # Canonical edge sets: sorted (min(u,v), max(u,v)) as tuples in sets
        def canonical_edges(edge_idx: torch.Tensor) -> set:
            src, tgt = edge_idx
            mins = torch.min(src, tgt)
            maxs = torch.max(src, tgt)
            return {(min.item(), max.item()) for min, max in zip(mins, maxs)}

        orig_canon = canonical_edges(orig.edge_index)
        cf_canon = canonical_edges(cf.edge_index)

        # Collect keep edge indices (keep original direction)
        keep_edges = []  # list of [2,1] tensors

        src_o, tgt_o = orig.edge_index
        for i in range(orig.num_edges):
            u, v = src_o[i].item(), tgt_o[i].item()
            edge_key = (min(u, v), max(u, v))

            # Rule: Skip if not in counterfactual
            if edge_key not in cf_canon:
                continue

            # Rule: Skip if both endpoints changed
            if u in changed_set and v in changed_set:
                continue

            # Retain
            keep_edges.append(orig.edge_index[:, i:i + 1])

        # Concat edges
        if keep_edges:
            keep_eidx = torch.cat(keep_edges, dim=1)
        else:
            keep_eidx = torch.empty((2, 0), dtype=torch.long, device=device)

        # Node features: original, zero changed
        non_explain_x = orig.x.clone()
        if len(changed_nodes) > 0:
            non_explain_x[changed_nodes] = 0

        return Data(x=non_explain_x, edge_index=keep_eidx)

    orig_list = _normalize_input(original)
    cf_list = _normalize_input(counterfactual)

    if len(orig_list) != len(cf_list):
        raise ValueError("Original and counterfactual inputs must have matching number of graphs.")

    non_explain_list = [_process_pair(o, c) for o, c in zip(orig_list, cf_list)]

    if len(non_explain_list) == 1:
        return non_explain_list[0]
    elif isinstance(original, Batch) or isinstance(counterfactual, Batch):
        return Batch.from_data_list(non_explain_list)
    else:
        return non_explain_list
