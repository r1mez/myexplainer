"""Evaluation utilities for MyExplainer."""

from typing import Dict, Tuple

import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from utils.batch_utils import output_to_batch
from utils.graph_utils import extract_explanatory_subgraph
from utils.vis_utils import visualize_explainer_graph


class _OracleWrappedGNN:
    """Simple GNN wrapper used for counting oracle calls when needed."""

    def __init__(self, gnn: torch.nn.Module):
        self.gnn = gnn
        self.oracle_calls = 0

    def get_pred(self, *args, **kwargs):
        self.oracle_calls += 1
        return self.gnn.get_pred(*args, **kwargs)

    def get_pred_explain(self, *args, **kwargs):
        self.oracle_calls += 1
        return self.gnn.get_pred_explain(*args, **kwargs)

    def eval(self):
        self.gnn.eval()


def _get_eval_graph_mode(args) -> str:
    return str(getattr(args, "eval_graph_mode", "discrete")).lower()


def _predict_cf_graphs(gnn, cf_graphs, eval_graph_mode: str):
    edge_weight = getattr(cf_graphs, "edge_weight", None)
    if eval_graph_mode == "continuous" and edge_weight is not None:
        return gnn.get_pred_explain(cf_graphs.x, cf_graphs.edge_index, edge_weight, cf_graphs.batch)
    return gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)


def _graph_to_undirected_edge_weight_map(graph: Data) -> Dict[Tuple[int, int], float]:
    edge_index = graph.edge_index
    if edge_index.numel() == 0:
        return {}

    edge_weight = getattr(graph, "edge_weight", None)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
    else:
        edge_weight = edge_weight.to(dtype=torch.float, device=edge_index.device)

    undirected_weights: Dict[Tuple[int, int], list] = {}
    src_all = edge_index[0].detach().cpu().tolist()
    dst_all = edge_index[1].detach().cpu().tolist()
    weights_all = edge_weight.detach().cpu().tolist()

    for src, dst, weight in zip(src_all, dst_all, weights_all):
        key = (int(src), int(dst))
        if key[0] > key[1]:
            key = (key[1], key[0])
        undirected_weights.setdefault(key, []).append(float(weight))

    return {
        key: sum(values) / max(len(values), 1)
        for key, values in undirected_weights.items()
    }


def _continuous_graph_deltas(ori_graph: Data, cf_graph: Data):
    ori_map = _graph_to_undirected_edge_weight_map(ori_graph)
    cf_map = _graph_to_undirected_edge_weight_map(cf_graph)
    union_keys = set(ori_map) | set(cf_map)
    soft_edit_mass = sum(abs(cf_map.get(key, 0.0) - ori_map.get(key, 0.0)) for key in union_keys)
    return soft_edit_mass, len(union_keys), len(ori_map)


def evaluate(args, model, gnn, data_loader):
    model.eval()

    wrapped_gnn = _OracleWrappedGNN(gnn)
    wrapped_gnn.eval()
    original_train_mode = getattr(args, "train_mode", False)
    args.train_mode = False
    eval_graph_mode = _get_eval_graph_mode(args)

    y_desired_all = []
    ori_prob_all = []
    ori_pred_all = []
    with torch.no_grad():
        for batch in data_loader:
            origraphs = batch["graphs"].to(args.device)
            _, ori_pred_logits = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
            ori_prob = F.softmax(ori_pred_logits, dim=1)
            ori_pred = ori_pred_logits.argmax(dim=1)
            y_desired = (1 - ori_pred).float().unsqueeze(1)
            y_desired_all.append(y_desired.cpu())
            ori_prob_all.append(ori_prob.cpu())
            ori_pred_all.append(ori_pred.cpu())

    proximity = 0.0
    valid_cf = 0
    fidel_sum = 0.0
    sparsity_sum = 0.0
    class_total = {0: 0, 1: 0}
    class_success = {0: 0, 1: 0}

    total = data_loader.dataset.__len__()
    total_cf_time = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating:")):
            origraphs = batch["graphs"].to(args.device)
            subgraphs = batch["subgraphs"]

            y_desired = y_desired_all[batch_idx].to(args.device)
            ori_pred = ori_pred_all[batch_idx].to(args.device).view(-1).long()

            t0 = time.time()
            outputs = model(
                graphs=origraphs,
                subgraphs=subgraphs,
                cond_labels=y_desired,
            )
            total_cf_time += time.time() - t0

            visualize_explainer_graph(
                origraphs,
                y_desired,
                outputs,
                dataset_name=args.dataset,
                render_mode=eval_graph_mode,
            )

            if eval_graph_mode == "continuous":
                cf_graphs = output_to_batch(origraphs, outputs, use_hard=False)
            else:
                cf_graphs = output_to_batch(origraphs, outputs, use_hard=True, thresh=0.5)

            if batch_idx == 0:
                ori_graphs_list = origraphs.to_data_list()
                cf_graphs_list = cf_graphs.to_data_list()

                print(f"\n[DEBUG] Batch {batch_idx} - First 3 graphs:")
                for i in range(min(3, len(ori_graphs_list))):
                    ori_edges = ori_graphs_list[i].num_edges
                    cf_edges = cf_graphs_list[i].num_edges
                    if eval_graph_mode == "continuous":
                        soft_edit_mass, _, ori_undirected_edges = _continuous_graph_deltas(
                            ori_graphs_list[i], cf_graphs_list[i]
                        )
                        soft_sparsity = 1.0 - soft_edit_mass / max(ori_undirected_edges, 1)
                        soft_sparsity = max(0.0, min(1.0, soft_sparsity))
                        print(
                            f"  Graph {i}: ori_edges={ori_edges}, cf_edges={cf_edges}, "
                            f"soft_edit_mass={soft_edit_mass:.4f}, soft_sparsity={soft_sparsity:.4f}"
                        )
                    else:
                        exp_graph = extract_explanatory_subgraph(ori_graphs_list[i], cf_graphs_list[i])
                        exp_edges = exp_graph.num_edges
                        print(f"  Graph {i}: ori_edges={ori_edges}, cf_edges={cf_edges}, exp_edges={exp_edges}")
                        print(f"            sparsity = 1 - ({exp_edges}/{ori_edges}) = {1 - exp_edges / ori_edges:.4f}")

            success_mask = get_flip_success_mask(
                y_desired,
                cf_graphs,
                gnn,
                eval_graph_mode=eval_graph_mode,
            )
            valid_cf += int(success_mask.sum().item())
            for class_idx in class_total:
                class_mask = ori_pred == class_idx
                class_total[class_idx] += int(class_mask.sum().item())
                if class_mask.any():
                    class_success[class_idx] += int(success_mask[class_mask].sum().item())
            proximity += compute_proximity(
                args,
                cf_graphs,
                origraphs,
                eval_graph_mode=eval_graph_mode,
            )
            fidel_sum += compute_fidelity_prob(
                args,
                origraphs,
                cf_graphs,
                ori_prob_all[batch_idx],
                gnn,
                eval_graph_mode=eval_graph_mode,
            )
            sparsity_sum += compute_sparsity(
                args,
                origraphs,
                cf_graphs,
                eval_graph_mode=eval_graph_mode,
            )

    validity = valid_cf / total if total > 0 else 0.0
    sparsity = sparsity_sum / total if total > 0 else 0.0
    avg_proximity = proximity / total if total > 0 else 0.0
    fidelity = fidel_sum / total if total > 0 else 0.0
    avg_runtime_per_graph = total_cf_time / total if total > 0 else 0.0
    avg_oracle_calls_per_graph = wrapped_gnn.oracle_calls / total if total > 0 else 0.0

    per_class_flip = {}
    for class_idx in class_total:
        total_class = class_total[class_idx]
        success_class = class_success[class_idx]
        per_class_flip[class_idx] = {
            "successful": success_class,
            "total": total_class,
            "ratio": success_class / total_class if total_class > 0 else 0.0,
        }

    args.train_mode = original_train_mode

    return {
        "validity": validity,
        "proximity": avg_proximity,
        "fidelity": fidelity,
        "sparsity": sparsity,
        "successful": valid_cf,
        "total": total,
        "runtime": avg_runtime_per_graph,
        "oracle_calls": avg_oracle_calls_per_graph,
        "per_class_flip": per_class_flip,
    }


def get_flip_success_mask(target_lables, cf_graphs, gnn, eval_graph_mode: str = "discrete"):
    gnn.eval()

    if eval_graph_mode == "continuous":
        _, pred_logits_cf = _predict_cf_graphs(gnn, cf_graphs, eval_graph_mode)
        pred_labels_cf = pred_logits_cf.argmax(dim=1).view(-1, 1)
    else:
        pred_logits_cf = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
        pred_labels_cf = pred_logits_cf.argmax(dim=1).view(-1, 1)
    return (pred_labels_cf == target_lables).view(-1)


def count_valid(target_lables, cf_graphs, gnn, eval_graph_mode: str = "discrete"):
    flipped_lables = get_flip_success_mask(
        target_lables,
        cf_graphs,
        gnn,
        eval_graph_mode=eval_graph_mode,
    ).sum().item()
    return flipped_lables


def compute_proximity(args, cf_graphs, ori_graphs, eval_graph_mode: str = "discrete"):
    if eval_graph_mode == "continuous":
        ori_graphs_list = ori_graphs.to_data_list()
        cf_graphs_list = cf_graphs.to_data_list()
        proximity = 0.0
        for ori_graph, cf_graph in zip(ori_graphs_list, cf_graphs_list):
            soft_edit_mass, union_size, _ = _continuous_graph_deltas(ori_graph, cf_graph)
            proximity += soft_edit_mass / max(union_size, 1)
        return proximity

    rho = 1.0
    ori_graphs_list = ori_graphs.to_data_list()
    cf_graphs_list = cf_graphs.to_data_list()
    batch_size = len(ori_graphs_list)
    distances = torch.zeros(batch_size, device=args.device)

    for i in range(batch_size):
        orig_data = ori_graphs_list[i]
        cf_data = cf_graphs_list[i]

        if getattr(orig_data, "num_nodes", None) is not None:
            num_nodes = orig_data.num_nodes
        elif getattr(orig_data, "x", None) is not None:
            num_nodes = orig_data.x.size(0)
        else:
            max_idx = 0
            if orig_data.edge_index.numel() > 0:
                max_idx = int(orig_data.edge_index.max())
            if cf_data.edge_index.numel() > 0:
                max_idx = max(max_idx, int(cf_data.edge_index.max()))
            num_nodes = max_idx + 1

        orig_adj = (
            to_dense_adj(orig_data.edge_index, max_num_nodes=num_nodes).squeeze(0)
            if orig_data.edge_index.numel() > 0
            else torch.zeros(num_nodes, num_nodes, device=args.device)
        )
        cf_adj = (
            to_dense_adj(cf_data.edge_index, max_num_nodes=num_nodes).squeeze(0)
            if cf_data.edge_index.numel() > 0
            else torch.zeros(num_nodes, num_nodes, device=args.device)
        )

        d_adj_entries = torch.norm(orig_adj - cf_adj, p=1)

        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges
        max_m = max(m_orig, m_cf)
        normalization = 2.0 * max_m if max_m > 0 else 1.0

        distances[i] = rho * (d_adj_entries / normalization)

    return distances.sum().item()


def compute_fidelity_prob(args, ori_graphs, cf_graphs, ori_prob, gnn, eval_graph_mode: str = "discrete"):
    ori_pred = ori_prob.argmax(dim=1)

    if eval_graph_mode == "continuous":
        cf_prob, _ = _predict_cf_graphs(gnn, cf_graphs, eval_graph_mode)
    else:
        cf_pred_logits = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
        cf_prob = F.softmax(cf_pred_logits, dim=1)

    fidelity_sum = 0.0
    for i in range(len(ori_pred)):
        ori_prob_single = ori_prob[i, ori_pred[i]].item()
        cf_prob_single = cf_prob[i, ori_pred[i]].item()
        fidelity_sum += ori_prob_single - cf_prob_single

    return fidelity_sum


def compute_sparsity(args, ori_graphs, cf_graphs, eval_graph_mode: str = "discrete"):
    if eval_graph_mode == "continuous":
        ori_graphs_list = ori_graphs.to_data_list()
        cf_graphs_list = cf_graphs.to_data_list()
        sparsity = 0.0
        for ori_graph, cf_graph in zip(ori_graphs_list, cf_graphs_list):
            soft_edit_mass, _, ori_undirected_edges = _continuous_graph_deltas(ori_graph, cf_graph)
            soft_sparsity = 1.0 - soft_edit_mass / max(ori_undirected_edges, 1)
            sparsity += max(0.0, min(1.0, soft_sparsity))
        return sparsity

    ori_graphs_list = ori_graphs.to_data_list()
    cf_graphs_list = cf_graphs.to_data_list()
    exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs_list, cf_graphs_list)]

    exp_num_edges = [exp.num_edges for exp in exp_graphs]
    ori_num_edges = [ori.num_edges for ori in ori_graphs_list]

    sparsity = 0.0
    for ori_edges, exp_edges in zip(ori_num_edges, exp_num_edges):
        sparsity += 1 - (exp_edges / ori_edges)

    return sparsity
