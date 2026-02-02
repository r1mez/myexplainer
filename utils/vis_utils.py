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
def visualize_subgraph(data, edge_mask):
    """
        根据给定的 mask 和 Data 对象，绘制原分子图并把mask部分加粗

        参数：
            data: torch_geometric.data.Data 对象，其中成员的 x 和 edge_attr 为 one-hot 编码
            edge_mask: data的子图mask，以torch.tensor存储
        """
    atom_map = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K',
                12: 'Li', 13: 'Ca'}
    bond_map = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}

    # 提取原子类型
    atom_indices = torch.argmax(data.x, dim=1).tolist()
    atoms = [atom_map[idx] for idx in atom_indices]

    # 创建可编辑分子
    editable_mol = Chem.RWMol()

    # 添加原子
    for atom_symbol in atoms:
        atom = Chem.Atom(atom_symbol)
        editable_mol.AddAtom(atom)

    # 提取边和键类型
    edge_index = data.edge_index.t().tolist()  # [[source, target], ...]
    bond_indices = torch.argmax(data.edge_attr, dim=1).tolist()
    bonds = [bond_map[idx] for idx in bond_indices]

    # 使用集合来跟踪已添加的键（避免重复）
    added_bonds = set()
    highlighted_bonds = []

    for i, (source, target) in enumerate(edge_index):
        # 标准化键顺序，确保 source < target 以避免重复
        bond_key = tuple(sorted([source, target]))
        if bond_key not in added_bonds:
            bond_type = bonds[i]
            editable_mol.AddBond(min(source, target), max(source, target), bond_type)
            added_bonds.add(bond_key)

            # 获取当前键的索引（添加后的键数量 - 1）
            bond_idx = editable_mol.GetMol().GetNumBonds() - 1

            # 检查 mask：由于双向边，检查当前 i 和反向边的 mask（如果存在）
            # 找到反向边的索引
            reverse_idx = None
            for j, (s, t) in enumerate(edge_index):
                if s == target and t == source:
                    reverse_idx = j
                    break

            # 如果当前或反向边的 mask 为 True，则高亮
            if edge_mask[i] or (reverse_idx is not None and edge_mask[reverse_idx]):
                highlighted_bonds.append(bond_idx)

    # 获取最终分子
    mol = editable_mol.GetMol()

    # 清理分子（推荐）
    # Chem.SanitizeMol(mol)
    _sanitize_with_valence_correction(mol)
    # mol = Chem.RemoveHs(mol,sanitize=False)

    # 绘制分子，高亮 mask 部分的键（加粗，使用红色高亮）
    img = MolToImage(mol, size=(600, 600), highlightBonds=highlighted_bonds, highlightColor=(0.8, 0.8, 0.8))
    img.save('molecule.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return img


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
# def visualize_explainer_graph(
#     graphs,
#     y_desired,
#     outputs,
#     g_idx: int = 5,
#     tau_add: float = 0.5,
#     layout: str = "spring",
#     seed: int = 42,
# ):
#     """
#     可视化一个 batch 中第 g_idx 张图的解释结果：
#       - 节点颜色: 统一颜色（原基于fs_node_mask的颜色映射已移除）
#       - 蓝色实线边: 原图边，颜色深浅 = adj_recon 对应的保留概率
#       - 红色虚线边: 原图不存在但 adj_recon > tau_add 的潜在新增边
#
#     Args:
#         graphs: PyG Batch, 包含 x, edge_index, batch 等，来自 train_loader 的 'graphs'
#         outputs: MyExplainerV2.forward(...) 的返回 dict（至少包含 adj_recon）
#         g_idx: 想看的第几张图（batch 中的图索引）
#         tau_add: 判定“新增边”的阈值，原来没边且 adj_recon > tau_add 的会画成红色虚线
#         layout: 节点布局，'spring' / 'kamada_kawai' / 'random' 等
#         seed: 随机种子保证布局可复现
#     """
#
#     device = graphs.x.device
#
#     x_all        = graphs.x                     # [N_total, x_dim]
#     edge_index   = graphs.edge_index            # [2, E_total]
#     batch        = graphs.batch                 # [N_total]
#     adj_recon    = outputs["adj_recon"]         # [N_total, N_total]
#
#     # ========== 1. 取出第 g_idx 张图的节点 ==========
#     node_idx = (batch == g_idx).nonzero(as_tuple=False).view(-1)  # [n_g]
#     if node_idx.numel() == 0:
#         print(f"[visualize_explainer_graph] batch 中没有索引为 {g_idx} 的图")
#         return
#
#     n_g = node_idx.size(0)
#     x_g = x_all[node_idx]                           # [n_g, x_dim]
#     adj_recon_g = adj_recon[node_idx][:, node_idx]  # [n_g, n_g]
#
#     # ========== 2. 构造这一图的原始边（局部索引） ==========
#     row_all, col_all = edge_index
#     edge_mask_g = (batch[row_all] == g_idx) & (batch[col_all] == g_idx)
#     edge_index_g_global = edge_index[:, edge_mask_g]   # [2, E_g_global]
#
#     # 全局 -> 局部 节点编号映射
#     global_to_local = {int(n.item()): i for i, n in enumerate(node_idx)}
#
#     orig_edges = []  # [(u_local, v_local), ...]
#     for i, j in edge_index_g_global.t():
#         gi, gj = int(i.item()), int(j.item())
#         if gi in global_to_local and gj in global_to_local:
#             u = global_to_local[gi]
#             v = global_to_local[gj]
#             if u != v:
#                 orig_edges.append((u, v))
#
#     orig_edges = list(set(tuple(sorted(e)) for e in orig_edges))  # 去重、无向
#
#     # 用 orig_edges 构造这一图的原始邻接标签 adj_label_g
#     adj_label_g = torch.zeros(n_g, n_g, device=device)
#     for u, v in orig_edges:
#         adj_label_g[u, v] = 1.0
#         adj_label_g[v, u] = 1.0
#
#     # ========== 3. 计算原始边的保留概率 & 潜在新增边 ==========
#     # 3.1 原图边的 adj_recon 分数
#     orig_scores = [float(adj_recon_g[u, v]) for (u, v) in orig_edges] if orig_edges else []
#
#     # 3.2 潜在新增边：原本没有边(adj_label_g=0)，但 adj_recon_g > tau_add
#     new_edges = []
#     new_scores = []
#     if tau_add is not None:
#         for i in range(n_g):
#             for j in range(i + 1, n_g):
#                 if adj_label_g[i, j] == 0 and float(adj_recon_g[i, j]) > tau_add:
#                     new_edges.append((i, j))
#                     new_scores.append(float(adj_recon_g[i, j]))
#
#     # ========== 4. 构造单图 Data & NetworkX 图 ==========
#     # 注意 edge_index 只用原始边，新增边只用来画图，不修改拓扑
#     if len(orig_edges) > 0:
#         edge_index_g_local = torch.tensor(orig_edges, dtype=torch.long).t().contiguous()
#     else:
#         edge_index_g_local = torch.empty((2, 0), dtype=torch.long)
#
#     data_g = Data(
#         x=x_g.cpu(),
#         edge_index=edge_index_g_local.cpu()
#     )
#     G = to_networkx(data_g, to_undirected=True)
#
#     if layout == "spring":
#         pos = nx.spring_layout(G, seed=seed)
#     elif layout == "kamada_kawai":
#         pos = nx.kamada_kawai_layout(G)
#     elif layout == "random":
#         pos = nx.random_layout(G)
#     else:
#         pos = nx.spring_layout(G, seed=seed)
#
#     # ========== 5. 开始画图 ==========
#     plt.figure(figsize=(6, 6))
#
#     # ---- 5.1 画节点：使用统一颜色（移除了基于fs_node_mask的颜色映射） ----
#     nodes = nx.draw_networkx_nodes(
#         G, pos,
#         node_color='lightblue',  # 统一节点颜色
#         node_size=500,           # 节点大小（可根据需要调整）
#         edgecolors='black'       # 节点边框，增强可读性
#     )
#
#     # ---- 5.2 画原始边：根据保留概率决定实线 / 虚线 ----
#     if len(orig_edges) > 0:
#         orig_scores_np = np.array(orig_scores)
#         norm_orig = plt.Normalize(vmin=orig_scores_np.min(), vmax=orig_scores_np.max())
#         orig_colors_rgba = plt.cm.Blues(norm_orig(orig_scores_np))
#
#         solid_edges = []
#         solid_colors = []
#         dashed_edges = []
#         dashed_colors = []
#
#         # 按 0.5 阈值把原始边分成“保留”(实线) 和 “弱保留”(虚线)
#         for edge, color, score in zip(orig_edges, orig_colors_rgba, orig_scores_np):
#             if score >= 0.5:
#                 solid_edges.append(edge)
#                 solid_colors.append(color)
#             else:
#                 dashed_edges.append(edge)
#                 dashed_colors.append(color)
#
#         # 概率高的边：蓝色实线
#         if solid_edges:
#             nx.draw_networkx_edges(
#                 G, pos,
#                 edgelist=solid_edges,
#                 edge_color=solid_colors,
#                 width=2.0,
#             )
#
#         # 概率低于 0.5 的边：蓝色虚线
#         if dashed_edges:
#             nx.draw_networkx_edges(
#                 G, pos,
#                 edgelist=dashed_edges,
#                 edge_color=dashed_colors,
#                 width=2.0,
#                 style="dashed",
#             )
#     # ---- 5.3 画潜在新增边：红色虚线，颜色深浅 = 新增概率 ----
#     if len(new_edges) > 0:
#         new_scores_np = np.array(new_scores)
#         norm_new = plt.Normalize(vmin=new_scores_np.min(), vmax=new_scores_np.max())
#         new_colors_rgba = plt.cm.Reds(norm_new(new_scores_np))
#
#         nx.draw_networkx_edges(
#             G, pos,
#             edgelist=new_edges,
#             edge_color=new_colors_rgba,
#             width=1.5,
#             style="dashed",
#         )
#
#     # ---- 5.4 标签 & colorbar ----
#     nx.draw_networkx_labels(G, pos, font_size=8)
#
#     # 原始边分数 colorbar（蓝）
#     if len(orig_edges) > 0:
#         sm_orig = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm_orig)
#         sm_orig.set_array([])
#         cb_orig = plt.colorbar(sm_orig, shrink=0.6, pad=0.02)
#         cb_orig.set_label("edge_score (adj_recon on original edges)", fontsize=10)
#
#     plt.title(
#         f"Graph #{g_idx} explanation (tau_add={tau_add}), y_hat={1 - y_desired[g_idx].item()},y_desired={y_desired[g_idx].item()}")
#     # plt.title(f"Graph #{g_idx} explanation (tau_add={tau_add}), y_hat={1-y_desired[g_idx].item()},y_desired={y_desired[g_idx].item()}\n{graphs[g_idx].smiles}")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# def visualize_explainer_graph(
#     graphs,
#     y_desired,
#     outputs,
#     g_idx: int = 5,
#     tau_add: float = 0.5,
#     layout: str = "spring",
#     seed: int = 42,
# ):
#     """
#     可视化一个 batch 中第 g_idx 张图的解释结果：
#       - 节点颜色: fs_node_mask (越红越重要)
#       - 蓝色实线边: 原图边，颜色深浅 = adj_recon 对应的保留概率
#       - 红色虚线边: 原图不存在但 adj_recon > tau_add 的潜在新增边
#
#     Args:
#         graphs: PyG Batch, 包含 x, edge_index, batch 等，来自 train_loader 的 'graphs'
#         outputs: MyExplainerV2.forward(...) 的返回 dict（至少包含 adj_recon, fs_node_mask）
#         g_idx: 想看的第几张图（batch 中的图索引）
#         tau_add: 判定“新增边”的阈值，原来没边且 adj_recon > tau_add 的会画成红色虚线
#         layout: 节点布局，'spring' / 'kamada_kawai' / 'random' 等
#         seed: 随机种子保证布局可复现
#     """
#
#     device = graphs.x.device
#
#     x_all        = graphs.x                     # [N_total, x_dim]
#     edge_index   = graphs.edge_index            # [2, E_total]
#     batch        = graphs.batch                 # [N_total]
#     adj_recon    = outputs["adj_recon"]         # [N_total, N_total]
#     fs_node_mask = outputs["fs_node_mask"]      # [N_total, 1]
#
#     # ========== 1. 取出第 g_idx 张图的节点 ==========
#     node_idx = (batch == g_idx).nonzero(as_tuple=False).view(-1)  # [n_g]
#     if node_idx.numel() == 0:
#         print(f"[visualize_explainer_graph] batch 中没有索引为 {g_idx} 的图")
#         return
#
#     n_g = node_idx.size(0)
#     x_g = x_all[node_idx]                           # [n_g, x_dim]
#     node_mask_g = fs_node_mask[node_idx].view(-1)   # [n_g]
#     adj_recon_g = adj_recon[node_idx][:, node_idx]  # [n_g, n_g]
#
#     # ========== 2. 构造这一图的原始边（局部索引） ==========
#     row_all, col_all = edge_index
#     edge_mask_g = (batch[row_all] == g_idx) & (batch[col_all] == g_idx)
#     edge_index_g_global = edge_index[:, edge_mask_g]   # [2, E_g_global]
#
#     # 全局 -> 局部 节点编号映射
#     global_to_local = {int(n.item()): i for i, n in enumerate(node_idx)}
#
#     orig_edges = []  # [(u_local, v_local), ...]
#     for i, j in edge_index_g_global.t():
#         gi, gj = int(i.item()), int(j.item())
#         if gi in global_to_local and gj in global_to_local:
#             u = global_to_local[gi]
#             v = global_to_local[gj]
#             if u != v:
#                 orig_edges.append((u, v))
#
#     orig_edges = list(set(tuple(sorted(e)) for e in orig_edges))  # 去重、无向
#
#     # 用 orig_edges 构造这一图的原始邻接标签 adj_label_g
#     adj_label_g = torch.zeros(n_g, n_g, device=device)
#     for u, v in orig_edges:
#         adj_label_g[u, v] = 1.0
#         adj_label_g[v, u] = 1.0
#
#     # ========== 3. 计算原始边的保留概率 & 潜在新增边 ==========
#     # 3.1 原图边的 adj_recon 分数
#     orig_scores = [float(adj_recon_g[u, v]) for (u, v) in orig_edges] if orig_edges else []
#
#     # 3.2 潜在新增边：原本没有边(adj_label_g=0)，但 adj_recon_g > tau_add
#     new_edges = []
#     new_scores = []
#     if tau_add is not None:
#         for i in range(n_g):
#             for j in range(i + 1, n_g):
#                 if adj_label_g[i, j] == 0 and float(adj_recon_g[i, j]) > tau_add:
#                     new_edges.append((i, j))
#                     new_scores.append(float(adj_recon_g[i, j]))
#
#     # ========== 4. 构造单图 Data & NetworkX 图 ==========
#     # 注意 edge_index 只用原始边，新增边只用来画图，不修改拓扑
#     if len(orig_edges) > 0:
#         edge_index_g_local = torch.tensor(orig_edges, dtype=torch.long).t().contiguous()
#     else:
#         edge_index_g_local = torch.empty((2, 0), dtype=torch.long)
#
#     data_g = Data(
#         x=x_g.cpu(),
#         edge_index=edge_index_g_local.cpu()
#     )
#     G = to_networkx(data_g, to_undirected=True)
#
#     if layout == "spring":
#         pos = nx.spring_layout(G, seed=seed)
#     elif layout == "kamada_kawai":
#         pos = nx.kamada_kawai_layout(G)
#     elif layout == "random":
#         pos = nx.random_layout(G)
#     else:
#         pos = nx.spring_layout(G, seed=seed)
#
#     # ========== 5. 开始画图 ==========
#     plt.figure(figsize=(6, 6))
#
#     # ---- 5.1 画节点：颜色 = fs_node_mask ----
#     node_color_np = node_mask_g.detach().cpu().numpy().reshape(-1)
#     nodes = nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_color_np,
#         cmap=plt.cm.Reds,
#     )
#
#     # ---- 5.2 画原始边：根据保留概率决定实线 / 虚线 ----
#     if len(orig_edges) > 0:
#         orig_scores_np = np.array(orig_scores)
#         norm_orig = plt.Normalize(vmin=orig_scores_np.min(), vmax=orig_scores_np.max())
#         orig_colors_rgba = plt.cm.Blues(norm_orig(orig_scores_np))
#
#         solid_edges = []
#         solid_colors = []
#         dashed_edges = []
#         dashed_colors = []
#
#         # 按 0.5 阈值把原始边分成“保留”(实线) 和 “弱保留”(虚线)
#         for edge, color, score in zip(orig_edges, orig_colors_rgba, orig_scores_np):
#             if score >= 0.5:
#                 solid_edges.append(edge)
#                 solid_colors.append(color)
#             else:
#                 dashed_edges.append(edge)
#                 dashed_colors.append(color)
#
#         # 概率高的边：蓝色实线
#         if solid_edges:
#             nx.draw_networkx_edges(
#                 G, pos,
#                 edgelist=solid_edges,
#                 edge_color=solid_colors,
#                 width=2.0,
#             )
#
#         # 概率低于 0.5 的边：蓝色虚线
#         if dashed_edges:
#             nx.draw_networkx_edges(
#                 G, pos,
#                 edgelist=dashed_edges,
#                 edge_color=dashed_colors,
#                 width=2.0,
#                 style="dashed",
#             )
#     # ---- 5.3 画潜在新增边：红色虚线，颜色深浅 = 新增概率 ----
#     if len(new_edges) > 0:
#         new_scores_np = np.array(new_scores)
#         norm_new = plt.Normalize(vmin=new_scores_np.min(), vmax=new_scores_np.max())
#         new_colors_rgba = plt.cm.Reds(norm_new(new_scores_np))
#
#         nx.draw_networkx_edges(
#             G, pos,
#             edgelist=new_edges,
#             edge_color=new_colors_rgba,
#             width=1.5,
#             style="dashed",
#         )
#
#     # ---- 5.4 标签 & colorbar ----
#     nx.draw_networkx_labels(G, pos, font_size=8)
#
#     # 节点 mask 的 colorbar
#     cb_nodes = plt.colorbar(nodes, shrink=0.6, pad=0.02)
#     cb_nodes.set_label("fs_node_mask", fontsize=10)
#
#     # 原始边分数 colorbar（蓝）
#     if len(orig_edges) > 0:
#         sm_orig = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm_orig)
#         sm_orig.set_array([])
#         cb_orig = plt.colorbar(sm_orig, shrink=0.6, pad=0.02)
#         cb_orig.set_label("edge_score (adj_recon on original edges)", fontsize=10)
#
#     plt.title(
#         f"Graph #{g_idx} explanation (tau_add={tau_add}), y_hat={1 - y_desired[g_idx].item()},y_desired={y_desired[g_idx].item()}")
#     # plt.title(f"Graph #{g_idx} explanation (tau_add={tau_add}), y_hat={1-y_desired[g_idx].item()},y_desired={y_desired[g_idx].item()}\n{graphs[g_idx].smiles}")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()