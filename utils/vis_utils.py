import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from sklearn.cluster import KMeans
from utils.graph_utils import _sanitize_with_valence_correction

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def visualize_explainer_graph(
        graphs,
        y_desired,
        outputs,
        g_idx: int = 0,
        tau_keep: float = 0.5,
        tau_add: float = 0.5,
        layout: str = "spring",
        seed: int = 42,
):
    """
    可视化优化版：
    1. 解决边重叠：若候选边已存在于原图，则跳过，不重复绘制。
    2. 视觉分层：保留边(实线)、删除边(淡化点线)、新增边(红色虚线)。
    """
    # 确保关闭之前的图表，防止内存泄漏或重叠
    plt.close('all')

    device = graphs.x.device
    x_all = graphs.x
    edge_index = graphs.edge_index
    batch = graphs.batch

    # 获取输出
    p_keep_all = outputs["p_keep"]
    cand_src = outputs.get("cand_src", None)
    cand_dst = outputs.get("cand_dst", None)
    p_add_all = outputs.get("p_add", None)

    fs_nodes_bool = outputs.get("fs_nodes_bool", None)
    fs_node_mask = outputs.get("fs_node_mask", None)

    # ========= 1. 取出第 g_idx 张图 =========
    node_idx = (batch == g_idx).nonzero(as_tuple=False).view(-1)
    if node_idx.numel() == 0:
        print(f"[visualize] batch 中没有索引为 {g_idx} 的图")
        return

    n_g = node_idx.size(0)
    x_g = x_all[node_idx]

    # 全局 -> 局部 映射
    global_to_local = {int(n.item()): i for i, n in enumerate(node_idx)}

    # ========= 2. 处理原始边 (Existing Edges) =========
    row_all, col_all = edge_index
    edge_mask_g = (batch[row_all] == g_idx) & (batch[col_all] == g_idx)
    edge_index_g_global = edge_index[:, edge_mask_g]
    p_keep_g = p_keep_all[edge_mask_g]

    # 用于快速查找 "这条边是否已存在"，防止新增边重叠
    existing_edges_set = set()

    edge_score_dict = {}
    for (gi, gj), pk in zip(edge_index_g_global.t(), p_keep_g):
        gi, gj = int(gi.item()), int(gj.item())
        if gi not in global_to_local or gj not in global_to_local or gi == gj:
            continue

        u, v = global_to_local[gi], global_to_local[gj]
        if u > v: u, v = v, u  # 无向图标准化

        key = (u, v)
        existing_edges_set.add(key)  # 记录已存在边

        if key not in edge_score_dict:
            edge_score_dict[key] = []
        edge_score_dict[key].append(float(pk.item()))

    orig_edges = list(edge_score_dict.keys())
    orig_scores = [np.mean(edge_score_dict[e]) for e in orig_edges] if orig_edges else []

    # ========= 3. 处理新增边 (New Edges) - 关键修改 =========
    new_edges = []
    new_scores = []

    if cand_src is not None and p_add_all is not None and p_add_all.numel() > 0:
        # 移到 CPU 处理方便
        cand_src_cpu = cand_src.detach().cpu()
        cand_dst_cpu = cand_dst.detach().cpu()
        p_add_cpu = p_add_all.detach().cpu()
        batch_cpu = batch.cpu()

        # 筛选属于当前图 g_idx 的候选边
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
            if u > v: u, v = v, u
            key = (u, v)

            # 【核心去重逻辑】：如果原图里已经有了这条边，不要把它算作新增边！
            # 即使 VGAE 预测了它，它也属于 "p_keep" 的范畴，而不是 "p_add"
            if key in existing_edges_set:
                continue

            if key not in add_score_dict:
                add_score_dict[key] = []
            add_score_dict[key].append(float(pa.item()))

        new_edges = list(add_score_dict.keys())
        new_scores = [np.mean(add_score_dict[e]) for e in new_edges] if new_edges else []

    # ========= 4. 构建 NetworkX 图 =========
    # 我们只用原始边构建 G，这样 layout 是基于原图骨架的
    if len(orig_edges) > 0:
        edge_index_local = torch.tensor(orig_edges, dtype=torch.long).t()
    else:
        edge_index_local = torch.empty((2, 0), dtype=torch.long)

    data_g = Data(x=x_g.cpu(), edge_index=edge_index_local)
    G = to_networkx(data_g, to_undirected=True)

    # 如果有新增节点（孤立点在原图中无边），确保它们也在 G 里
    if G.number_of_nodes() < n_g:
        G.add_nodes_from(range(n_g))

    # 布局
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # ========= 5. 绘图 =========
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # --- 5.1 画节点 ---
    # 颜色值
    if fs_nodes_bool is not None:
        node_vals = fs_nodes_bool[node_idx].float().cpu().numpy()
    elif fs_node_mask is not None:
        node_vals = fs_node_mask[node_idx].view(-1).detach().cpu().numpy()
    else:
        node_vals = np.ones(n_g)

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_vals,
        cmap=plt.cm.Reds,
        edgecolors="#333333",  # 深灰色边框
        linewidths=1.0,
        node_size=300,
        alpha=0.9
    )

    # --- 5.2 画原始边 (分类：保留 vs 删除) ---
    if len(orig_edges) > 0:
        orig_scores_np = np.array(orig_scores)

        edges_keep = []  # 保留
        edges_drop = []  # 删除

        for i, score in enumerate(orig_scores_np):
            if score >= tau_keep:
                edges_keep.append(orig_edges[i])
            else:
                edges_drop.append(orig_edges[i])

        # 1. 保留的边：深蓝色实线，显眼
        if edges_keep:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_keep,
                edge_color="#1f77b4",  # 标准蓝
                width=2.5,
                alpha=0.8
            )

        # 2. 拟删除的边：灰色/淡蓝色点线，非常淡，表示"即将消失"
        if edges_drop:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_drop,
                edge_color="grey",
                width=1.5,
                style="dotted",
                alpha=0.4  # 透明度高一点，不要喧宾夺主
            )

    # --- 5.3 画新增边 ---
    # 只有当概率 > tau_add 才画出来
    if len(new_edges) > 0:
        new_scores_np = np.array(new_scores)
        edges_add = []
        scores_add = []

        for i, score in enumerate(new_scores_np):
            if score >= tau_add:
                edges_add.append(new_edges[i])
                scores_add.append(score)

        if edges_add:
            # 使用颜色映射表示置信度，或者统一用红色
            # 这里统一用红色虚线，表示"反事实添加"
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_add,
                edge_color="#d62728",  # 标准红
                width=2.5,
                style="dashed",
                alpha=0.9
            )

    # --- 5.4 标签 ---
    # 尝试画原子类型
    smiles = getattr(graphs[g_idx], 'smiles', None)

    # 简单推断原子类型逻辑 (假设前几维是 one-hot)
    ATOM_TYPES = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']
    labels = {}
    x_cpu = x_g.cpu().numpy()

    # 如果特征维度很小，可能是 one-hot
    if x_cpu.shape[1] >= len(ATOM_TYPES):
        # 简单的启发式检查：看是否主要是 0/1
        if (x_cpu.max() <= 1.0) and (x_cpu.min() >= 0.0):
            for i in range(n_g):
                feat = x_cpu[i, :len(ATOM_TYPES)]
                idx = np.argmax(feat)
                # 只有当最大值接近1时才认为是该原子
                if feat[idx] > 0.5:
                    labels[i] = ATOM_TYPES[idx] if idx < len(ATOM_TYPES) else "?"
                else:
                    labels[i] = str(i)

    if not labels:
        labels = {i: str(i) for i in range(n_g)}

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black", font_weight="bold")

    # --- 5.5 图例与标题 ---
    plt.title(
        f"Graph #{g_idx} Explanation\nTarget: {y_desired[g_idx].item() if y_desired.ndim > 0 else y_desired.item()}",
        fontsize=12)

    # 手动添加图例，比 colorbar 更直观
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=2.5, label=f'Keep (p>{tau_keep})'),
        Line2D([0], [0], color='grey', lw=1.5, linestyle=':', alpha=0.5, label=f'Drop (p<{tau_keep})'),
        Line2D([0], [0], color='#d62728', lw=2.5, linestyle='--', label=f'Add (p>{tau_add})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaaaa', markersize=10, label='FS Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffeeee', markersize=10, label='Normal Node'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.axis("off")
    plt.tight_layout()
    plt.show()