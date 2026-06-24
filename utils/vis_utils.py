import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.node_labels import infer_feature_mode, infer_node_labels_for_dataset


def _labels_for_visualized_graph(x_g, dataset_name=None):
    feature_rows = x_g.detach().cpu().tolist()
    feature_mode = infer_feature_mode(feature_rows)
    labels = infer_node_labels_for_dataset(
        feature_rows,
        dataset=dataset_name,
        feature_mode=feature_mode,
    )
    return {idx: label for idx, label in enumerate(labels)}


def visualize_explainer_graph(
    graphs,
    y_desired,
    outputs,
    dataset_name: Optional[str] = None,
    g_idx: int = 0,
    render_mode: str = "discrete",
    tau_keep: float = 0.5,
    tau_add: float = 0.5,
    layout: str = "spring",
    seed: int = 42,
):
    """
    Visualize the generated counterfactual edit proposal for one graph in a batch.

    Existing edges and candidate additions can be rendered either as
    thresholded discrete edits or continuous edge strengths.
    """
    plt.close("all")

    x_all = graphs.x
    edge_index = graphs.edge_index
    batch = graphs.batch

    p_keep_all = outputs["p_keep"]
    cand_src = outputs.get("cand_src")
    cand_dst = outputs.get("cand_dst")
    p_add_all = outputs.get("p_add")

    fs_nodes_bool = outputs.get("fs_nodes_bool")
    fs_node_mask = outputs.get("fs_node_mask")

    node_idx = (batch == g_idx).nonzero(as_tuple=False).view(-1)
    if node_idx.numel() == 0:
        print(f"[visualize] batch 中没有索引为 {g_idx} 的图")
        return

    n_g = node_idx.size(0)
    x_g = x_all[node_idx]
    global_to_local = {int(n.item()): i for i, n in enumerate(node_idx)}

    row_all, col_all = edge_index
    edge_mask_g = (batch[row_all] == g_idx) & (batch[col_all] == g_idx)
    edge_index_g_global = edge_index[:, edge_mask_g]
    p_keep_g = p_keep_all[edge_mask_g]

    existing_edges_set = set()
    edge_score_dict = {}
    for (gi, gj), pk in zip(edge_index_g_global.t(), p_keep_g):
        gi, gj = int(gi.item()), int(gj.item())
        if gi not in global_to_local or gj not in global_to_local or gi == gj:
            continue

        u, v = global_to_local[gi], global_to_local[gj]
        if u > v:
            u, v = v, u

        key = (u, v)
        existing_edges_set.add(key)
        edge_score_dict.setdefault(key, []).append(float(pk.item()))

    orig_edges = list(edge_score_dict.keys())
    orig_scores = [np.mean(edge_score_dict[e]) for e in orig_edges] if orig_edges else []

    new_edges = []
    new_scores = []
    if cand_src is not None and cand_dst is not None and p_add_all is not None and p_add_all.numel() > 0:
        cand_src_cpu = cand_src.detach().cpu()
        cand_dst_cpu = cand_dst.detach().cpu()
        p_add_cpu = p_add_all.detach().cpu()
        batch_cpu = batch.cpu()

        mask_add_g = (batch_cpu[cand_src_cpu] == g_idx) & (batch_cpu[cand_dst_cpu] == g_idx)
        cand_src_g = cand_src_cpu[mask_add_g]
        cand_dst_g = cand_dst_cpu[mask_add_g]
        p_add_g = p_add_cpu[mask_add_g]

        add_score_dict = {}
        for gi, gj, pa in zip(cand_src_g, cand_dst_g, p_add_g):
            gi, gj = int(gi.item()), int(gj.item())
            if gi not in global_to_local or gj not in global_to_local or gi == gj:
                continue

            u, v = global_to_local[gi], global_to_local[gj]
            if u > v:
                u, v = v, u
            key = (u, v)
            if key in existing_edges_set:
                continue
            add_score_dict.setdefault(key, []).append(float(pa.item()))

        new_edges = list(add_score_dict.keys())
        new_scores = [np.mean(add_score_dict[e]) for e in new_edges] if new_edges else []

    if orig_edges:
        edge_index_local = torch.tensor(orig_edges, dtype=torch.long).t()
    else:
        edge_index_local = torch.empty((2, 0), dtype=torch.long)

    data_g = Data(x=x_g.cpu(), edge_index=edge_index_local)
    G = to_networkx(data_g, to_undirected=True)
    if G.number_of_nodes() < n_g:
        G.add_nodes_from(range(n_g))

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(10, 8))

    if fs_nodes_bool is not None:
        node_vals = fs_nodes_bool[node_idx].float().cpu().numpy()
    elif fs_node_mask is not None:
        node_vals = fs_node_mask[node_idx].view(-1).detach().cpu().numpy()
    else:
        node_vals = np.ones(n_g)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_vals,
        cmap=plt.cm.Reds,
        edgecolors="#333333",
        linewidths=1.0,
        node_size=300,
        alpha=0.9,
    )

    render_mode = str(render_mode).lower()

    if orig_edges and render_mode == "discrete":
        orig_scores_np = np.array(orig_scores)
        edges_keep = []
        edges_drop = []
        for i, score in enumerate(orig_scores_np):
            if score >= tau_keep:
                edges_keep.append(orig_edges[i])
            else:
                edges_drop.append(orig_edges[i])

        if edges_keep:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_keep,
                edge_color="#1f77b4",
                width=2.5,
                alpha=0.8,
            )

        if edges_drop:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_drop,
                edge_color="grey",
                width=1.5,
                style="dotted",
                alpha=0.4,
            )

    if new_edges and render_mode == "discrete":
        new_scores_np = np.array(new_scores)
        edges_add = []
        for i, score in enumerate(new_scores_np):
            if score >= tau_add:
                edges_add.append(new_edges[i])

        if edges_add:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_add,
                edge_color="#d62728",
                width=2.5,
                style="dashed",
                alpha=0.9,
            )

    if render_mode == "continuous":
        if orig_edges:
            orig_scores_np = np.array(orig_scores)
            score_min = float(orig_scores_np.min()) if orig_scores_np.size > 0 else 0.0
            score_max = float(orig_scores_np.max()) if orig_scores_np.size > 0 else 1.0
            denom = max(score_max - score_min, 1e-8)
            for edge, score in zip(orig_edges, orig_scores_np):
                if orig_scores_np.size > 1 and (score_max - score_min) > 1e-8:
                    normalized = (float(score) - score_min) / denom
                else:
                    normalized = float(score)
                width = 1.0 + 3.5 * max(float(score), 0.0)
                alpha = 0.2 + 0.75 * max(0.0, min(1.0, normalized))
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[edge],
                    edge_color="#1f77b4",
                    width=width,
                    alpha=alpha,
                )

        if new_edges:
            new_scores_np = np.array(new_scores)
            score_min = float(new_scores_np.min()) if new_scores_np.size > 0 else 0.0
            score_max = float(new_scores_np.max()) if new_scores_np.size > 0 else 1.0
            denom = max(score_max - score_min, 1e-8)
            for edge, score in zip(new_edges, new_scores_np):
                if new_scores_np.size > 1 and (score_max - score_min) > 1e-8:
                    normalized = (float(score) - score_min) / denom
                else:
                    normalized = float(score)
                width = 1.0 + 3.5 * max(float(score), 0.0)
                alpha = 0.2 + 0.75 * max(0.0, min(1.0, normalized))
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[edge],
                    edge_color="#d62728",
                    width=width,
                    style="dashed",
                    alpha=alpha,
                )

    labels = _labels_for_visualized_graph(x_g, dataset_name=dataset_name)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black", font_weight="bold")

    plt.title(
        f"Graph #{g_idx} Explanation\nTarget: {y_desired[g_idx].item() if y_desired.ndim > 0 else y_desired.item()}",
        fontsize=12,
    )

    from matplotlib.lines import Line2D

    if render_mode == "continuous":
        legend_elements = [
            Line2D([0], [0], color="#1f77b4", lw=2.5, alpha=0.8, label="Keep strength"),
            Line2D([0], [0], color="#d62728", lw=2.5, linestyle="--", alpha=0.8, label="Add strength"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffaaaa", markersize=10, label="FS Node"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffeeee", markersize=10, label="Normal Node"),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color="#1f77b4", lw=2.5, label=f"Keep (p>{tau_keep})"),
            Line2D([0], [0], color="grey", lw=1.5, linestyle=":", alpha=0.5, label=f"Drop (p<{tau_keep})"),
            Line2D([0], [0], color="#d62728", lw=2.5, linestyle="--", label=f"Add (p>{tau_add})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffaaaa", markersize=10, label="FS Node"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffeeee", markersize=10, label="Normal Node"),
        ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
