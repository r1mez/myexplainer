import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


def compute_proximity_from_edge_index(
    ori_edge_index: torch.Tensor,
    cf_edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> float:
    """
    按照 MyExplainer 的定义计算单个图的 Proximity。
    使用 L1 范数并按 2 * max_m 归一化，等价于 evaluationV2.compute_proximity
    中对每个图的处理逻辑。
    """
    ori_edge_index = ori_edge_index.to(device)
    cf_edge_index = cf_edge_index.to(device)

    # 邻接矩阵，强制统一为 [N, N]
    ori_adj = to_dense_adj(ori_edge_index, max_num_nodes=num_nodes).squeeze(0)
    cf_adj = to_dense_adj(cf_edge_index, max_num_nodes=num_nodes).squeeze(0)

    # L1 范数，统计矩阵条目的变化量
    d_adj_entries = torch.norm(ori_adj - cf_adj, p=1)

    # 归一化：2 * max_m，其中 max_m 是无向图边数（或有向图边数）
    m_ori = ori_edge_index.size(1) // 2
    m_cf = cf_edge_index.size(1) // 2
    max_m = max(m_ori, m_cf)
    normalization = 2.0 * max_m if max_m > 0 else 1.0

    return (d_adj_entries / normalization).item()


def compute_fidelity_prob_from_probs(
    ori_probs: torch.Tensor,
    cf_probs: torch.Tensor,
) -> float:
    """
    按照 MyExplainer 的定义计算单个图的 Fidelity（概率版）。
    等价于 evaluationV2.compute_fidelity_prob 中：
    ori_prob[ori_pred] - cf_prob[ori_pred]。
    """
    # 输入假定是一维 [num_classes]
    if ori_probs.dim() > 1:
        ori_probs = ori_probs.squeeze(0)
    if cf_probs.dim() > 1:
        cf_probs = cf_probs.squeeze(0)

    ori_pred = ori_probs.argmax().item()
    return (ori_probs[ori_pred] - cf_probs[ori_pred]).item()


def compute_sparsity_from_edge_index(
    ori_edge_index: torch.Tensor,
    cf_edge_index: torch.Tensor,
) -> float:
    """
    按照 MyExplainer 的稀疏性含义计算单个图的 Sparsity。
    evaluationV2 通过 extract_explanatory_subgraph 得到解释子图，
    其边数本质上等价于原图与 CF 图边集的对称差大小。
    这里直接用边集对称差来近似：
        sparsity = 1 - (#changed_edges / #ori_edges)
    """
    ori_edge_index = ori_edge_index.cpu()
    cf_edge_index = cf_edge_index.cpu()

    ori_set = set(
        (min(u, v), max(u, v)) for u, v in ori_edge_index.t().tolist()
    )
    cf_set = set(
        (min(u, v), max(u, v)) for u, v in cf_edge_index.t().tolist()
    )

    if len(ori_set) == 0:
        return 0.0

    diff = ori_set.symmetric_difference(cf_set)
    return 1.0 - len(diff) / max(len(ori_set), 1)

