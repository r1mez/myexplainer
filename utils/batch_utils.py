import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj


def core_data_from_batch(args, batch):
    """
    从批次数据中提取子图的节点特征和边属性

    参数:
        args: 包含配置参数的对象，需要有max_subgraph_nodes, x_dim, edge_attr_dim等属性
        batch: 包含子图数据的批次对象

    返回:
        tuple: (all_subgraph_x, all_subgraph_adj, all_subgraph_edge_attr)
            - all_subgraph_x: 节点特征张量，形状为[B, max_subgraph_nodes, x_dim]
            - all_subgraph_adj: 邻接矩阵，形状为[B, max_subgraph_nodes, max_subgraph_nodes]
            - all_subgraph_edge_attr: 边属性张量，形状为[B, M, edge_attr_dim]，
              其中M是下三角矩阵的边数
    """

    # 第一步：提取子图节点特征
    actual_B = batch["subgraphs"].batch.max().item() + 1

    subgraph_x_list = []
    zero_x_template = torch.zeros(args.max_subgraph_nodes, args.x_dim, device=args.device)
    for b in range(batch["graphs"].num_graphs):
        # 提取单个子图
        mask = (batch['subgraphs'].batch == b)
        num_nodes_tensor = mask.sum()  # 保持为张量，避免 .item() 过早转换

        subgraph_x = batch['subgraphs'].x[mask]  # 提取该子图的节点特征

        # 使用clone而不是zeros创建新张量
        subgraph_x_padded = zero_x_template.clone()
        num_nodes_i = num_nodes_tensor.item()  # 只在赋值时转换
        if num_nodes_i > 0:  # 避免空图
            subgraph_x_padded = torch.zeros((subgraph_x.size(0), subgraph_x.size(1)), dtype=subgraph_x.dtype)
            subgraph_x_padded[:num_nodes_i] = subgraph_x

        # 存储子图数据
        subgraph_x_list.append(subgraph_x_padded)

    all_subgraph_x = torch.stack(subgraph_x_list, dim=0)  # (B, max_num_nodes, x_dim)

    # 第二步：提取子图邻接矩阵和边特征
    all_subgraph_adj = to_dense_adj(
        batch["subgraphs"].edge_index,
        batch=batch["subgraphs"].batch,
        max_num_nodes=args.max_subgraph_nodes
    )
    all_subgraph_edge_attr_dense = to_dense_adj(
        batch["subgraphs"].edge_index,
        batch=batch["subgraphs"].batch,
        edge_attr=batch["subgraphs"].edge_attr,  # [total_edges, edge_attr_dim]，如果无则 None（会默认全1）
        max_num_nodes=args.max_subgraph_nodes
    )  # 形状: [B, N, N, edge_attr_dim]；缺失边填充0

    if args.edge_attr_dim != 0:
        # 提取下三角边特征（[B, M, edge_attr_dim]）
        N = args.max_subgraph_nodes
        tril_indices = torch.tril_indices(N, N, offset=-1,
                                          device=all_subgraph_edge_attr_dense.device)  # [2, M]：行/列索引
        M = tril_indices[0].numel()  # N*(N-1)//2

        all_subgraph_edge_attr = all_subgraph_edge_attr_dense[
                                 :, tril_indices[0], tril_indices[1], :
                                 ].reshape(actual_B, M, args.edge_attr_dim)  # [B, M, edge_attr_dim]；reshape 确保形状正确

        # 第三步：将数据移到设备上
        all_subgraph_x = all_subgraph_x.to(args.device)
        all_subgraph_adj = all_subgraph_adj.to(args.device)
        all_subgraph_edge_attr = all_subgraph_edge_attr.to(args.device)

        return all_subgraph_x, all_subgraph_adj, all_subgraph_edge_attr
    else:
        # 第三步：将数据移到设备上
        all_subgraph_x = all_subgraph_x.to(args.device)
        all_subgraph_adj = all_subgraph_adj.to(args.device)

        return all_subgraph_x, all_subgraph_adj, None


def output_to_batch(graphs, outputs, use_hard=True, thresh=0.5) -> Batch:
    x_all = graphs.x
    batch = graphs.batch
    y_all = getattr(graphs, 'y', None)

    edge_index_cf = outputs["edge_index_cf"]          # [2, E_cf]
    edge_weight_cf = outputs.get("edge_weight_cf", None)  # [E_cf] or None

    # 如果需要硬阈值，把权重小于 thresh 的边删掉
    if use_hard and edge_weight_cf is not None:
        mask = edge_weight_cf >= thresh
        edge_index_cf = edge_index_cf[:, mask]
        edge_weight_cf = edge_weight_cf[mask]

    B = int(batch.max().item()) + 1
    device = x_all.device

    data_list = []

    for g in range(B):
        # 这一图的所有节点（全局索引）
        node_mask = (batch == g)
        node_idx = node_mask.nonzero(as_tuple=False).view(-1)  # [n_g]
        if node_idx.numel() == 0:
            continue

        # 全局 idx -> 局部 idx 映射
        # 例如 node_idx = [10, 11, 13]，则 mapping[10]=0, mapping[11]=1, mapping[13]=2
        mapping = {int(n.item()): i for i, n in enumerate(node_idx)}

        # 在 cf 边中筛选出“两个端点都属于这一图”的边
        src_all = edge_index_cf[0]
        dst_all = edge_index_cf[1]
        edge_mask_g = node_mask[src_all] & node_mask[dst_all]
        e_idx_g = edge_index_cf[:, edge_mask_g]  # [2, E_g]

        if e_idx_g.size(1) > 0:
            src_global = e_idx_g[0]
            dst_global = e_idx_g[1]

            # 映射到局部 idx
            # 注意：这里用 list comprehension + dict，会比纯 tensor 操作慢一点，
            # 但图通常不大，代码清晰好调试
            src_local = torch.tensor(
                [mapping[int(i.item())] for i in src_global],
                dtype=torch.long,
                device=device,
            )
            dst_local = torch.tensor(
                [mapping[int(i.item())] for i in dst_global],
                dtype=torch.long,
                device=device,
            )

            edge_index_local = torch.stack([src_local, dst_local], dim=0)
            if edge_weight_cf is not None:
                edge_weight_local = edge_weight_cf[edge_mask_g]
            else:
                edge_weight_local = None
        else:
            # 这一图在 CF 中没有任何边（极端情况）
            edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weight_local = None

        # 节点特征
        x_g = x_all[node_idx]

        data_g = Data(
            x=x_g,
            edge_index=edge_index_local,
        )
        if edge_weight_local is not None:
            data_g.edge_weight = edge_weight_local

        if y_all is not None:
            # 如果原来是图级标签，就取这一图的 y
            # 这里假设 graphs.y 的 shape 是 [B] 或 [B, *]
            if y_all.dim() == 1:
                data_g.y = y_all[g]
            else:
                data_g.y = y_all[g:g+1]

        data_list.append(data_g)

    cf_batch = Batch.from_data_list(data_list)
    return cf_batch