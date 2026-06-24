import torch
from eval.metrics import proximity, fidelity, sparsity


class OracleWrappedModel(torch.nn.Module):
    """
    对 GNN 分类器的简单包装，用于在 baseline 评估中统计 oracle 调用次数。
    - 统计 forward 调用次数（model(x, edge_index, batch, ...)）
    - 统计 get_pred_explain 调用次数（model.get_pred_explain(...)）
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.oracle_calls = 0

    def forward(self, *args, **kwargs):
        """统计一次前向调用，并转发给原始模型。"""
        self.oracle_calls += 1
        return self.model(*args, **kwargs)

    def get_pred_explain(self, *args, **kwargs):
        """统计一次 get_pred_explain 调用，并转发给原始模型。"""
        self.oracle_calls += 1
        return self.model.get_pred_explain(*args, **kwargs)

    def __getattr__(self, name):
        # 避免递归地访问自身属性
        if name in {"model", "oracle_calls"}:
            return super().__getattr__(name)
        return getattr(self.model, name)


# Backward-compatible aliases for baseline code that uses the old raw-tensor API.
# These wrap the consolidated eval.metrics functions for single-graph usage.


def compute_proximity_from_edge_index(
    ori_edge_index: torch.Tensor,
    cf_edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> float:
    """
    Single-graph proximity from raw edge_index tensors.
    Delegates to eval.metrics.proximity via a temporary Data wrapper.
    """
    from torch_geometric.data import Batch, Data

    ori_data = Data(edge_index=ori_edge_index.to(device), num_nodes=num_nodes)
    cf_data = Data(edge_index=cf_edge_index.to(device), num_nodes=num_nodes)
    # is_undirected check requires the graph to have edges; add x for safety
    if ori_edge_index.numel() > 0:
        ori_data.x = torch.zeros(num_nodes, 1, device=device)
    else:
        ori_data.x = torch.zeros(num_nodes, 1, device=device)
    if cf_edge_index.numel() > 0:
        cf_data.x = torch.zeros(num_nodes, 1, device=device)
    else:
        cf_data.x = torch.zeros(num_nodes, 1, device=device)

    class _Config:
        def __init__(self, d):
            self.device = d

    ori_batch = Batch.from_data_list([ori_data])
    cf_batch = Batch.from_data_list([cf_data])
    return proximity(_Config(device), cf_batch, ori_batch)


def compute_fidelity_prob_from_probs(
    ori_probs: torch.Tensor,
    cf_probs: torch.Tensor,
) -> float:
    """
    Single-graph fidelity from pre-computed probability tensors.
    Computes ori_prob[ori_pred] - cf_prob[ori_pred].
    """
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
    Single-graph sparsity from raw edge_index tensors.
    Uses edge-set symmetric difference as proxy for explanatory subgraph.
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
