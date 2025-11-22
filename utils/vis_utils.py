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
    可视化 CFExplainerVGAEAdd 对第 g_idx 张图的解释结果：

      - 节点颜色: 由 fs_nodes_bool / fs_node_mask 决定（FS 节点更红）
      - 原始边:
          * 蓝色实线: p_keep >= tau_keep（保留的边）
          * 蓝色虚线: p_keep <  tau_keep（倾向删除的边）
      - 新增边:
          * 红色虚线: 候选新增边 (cand_src, cand_dst)，且 p_add >= tau_add

    Args:
        graphs: PyG Batch，包含 x, edge_index, batch 等
        y_desired: [B] 或 [B,1]，反事实目标标签，用于标题显示
        outputs: CFExplainerVGAEAdd.forward(...) 的输出 dict
        g_idx: 可视化 batch 中的第几张图
        tau_keep: 判定“保留 / 删除”原始边的概率阈值
        tau_add: 判定“画不画出来新增边”的概率阈值
        layout: 'spring' / 'kamada_kawai' / 'random'
        seed: spring 布局随机种子
    """
    device = graphs.x.device

    x_all = graphs.x                      # [N_total, x_dim]
    edge_index = graphs.edge_index        # [2, E_total]
    batch = graphs.batch                  # [N_total]

    # 来自 CFExplainerVGAEAdd.forward 的输出
    p_keep_all = outputs["p_keep"]        # [E_total]
    cand_src = outputs.get("cand_src", None)   # [M_add] or None
    cand_dst = outputs.get("cand_dst", None)   # [M_add] or None
    p_add_all = outputs.get("p_add", None)     # [M_add] or None

    fs_nodes_bool = outputs.get("fs_nodes_bool", None)   # [N_total] bool
    fs_node_mask = outputs.get("fs_node_mask", None)     # [N_total,1] or [N_total]

    # ========= 1. 取出第 g_idx 张图的节点 =========
    node_idx = (batch == g_idx).nonzero(as_tuple=False).view(-1)  # [n_g]
    if node_idx.numel() == 0:
        print(f"[visualize_explainer_graph] batch 中没有索引为 {g_idx} 的图")
        return

    n_g = node_idx.size(0)
    x_g = x_all[node_idx]  # [n_g, x_dim]

    # ========= 2. 节点颜色：FS 节点更红 =========
    if fs_nodes_bool is not None:
        node_vals = fs_nodes_bool[node_idx].float().cpu().numpy()
    elif fs_node_mask is not None:
        node_vals = fs_node_mask[node_idx].view(-1).detach().cpu().numpy()
    else:
        node_vals = np.ones(n_g, dtype=np.float32)

    # ========= 3. 构造这一图的原始边 + p_keep =========
    row_all, col_all = edge_index
    # 只保留属于 g_idx 这张图的边
    edge_mask_g = (batch[row_all] == g_idx) & (batch[col_all] == g_idx)
    edge_index_g_global = edge_index[:, edge_mask_g]      # [2, E_g]
    p_keep_g = p_keep_all[edge_mask_g]                    # [E_g]

    # 全局 -> 局部 节点编号映射
    global_to_local = {int(n.item()): i for i, n in enumerate(node_idx)}

    # 收集无向边 (u_local, v_local) 以及对应的 p_keep（注意去重）
    edge_score_dict = {}  # key: (u,v), val: [scores]
    for (gi, gj), pk in zip(edge_index_g_global.t(), p_keep_g):
        gi = int(gi.item())
        gj = int(gj.item())
        if gi not in global_to_local or gj not in global_to_local or gi == gj:
            continue
        u = global_to_local[gi]
        v = global_to_local[gj]
        if u > v:
            u, v = v, u
        key = (u, v)
        if key not in edge_score_dict:
            edge_score_dict[key] = []
        edge_score_dict[key].append(float(pk.item()))

    orig_edges = list(edge_score_dict.keys())
    orig_scores = [np.mean(edge_score_dict[e]) for e in orig_edges] if orig_edges else []

    # ========= 4. 构造这一图的候选新增边 + p_add =========
    new_edges = []
    new_scores = []

    if cand_src is not None and p_add_all is not None and p_add_all.numel() > 0:
        cand_src = cand_src.to(device)
        cand_dst = cand_dst.to(device)
        p_add_all = p_add_all.to(device)

        # 只保留当前图 g_idx 内部的候选边
        mask_add_g = (batch[cand_src] == g_idx) & (batch[cand_dst] == g_idx)
        cand_src_g = cand_src[mask_add_g]
        cand_dst_g = cand_dst[mask_add_g]
        p_add_g = p_add_all[mask_add_g]

        add_score_dict = {}
        for gi, gj, pa in zip(cand_src_g, cand_dst_g, p_add_g):
            gi = int(gi.item())
            gj = int(gj.item())
            if gi not in global_to_local or gj not in global_to_local or gi == gj:
                continue
            u = global_to_local[gi]
            v = global_to_local[gj]
            if u > v:
                u, v = v, u
            key = (u, v)
            if key not in add_score_dict:
                add_score_dict[key] = []
            add_score_dict[key].append(float(pa.item()))

        new_edges = list(add_score_dict.keys())
        new_scores = [np.mean(add_score_dict[e]) for e in new_edges] if new_edges else []

    # ========= 5. 构造单图 Data & NetworkX 图 =========
    if len(orig_edges) > 0:
        edge_index_g_local = torch.tensor(orig_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index_g_local = torch.empty((2, 0), dtype=torch.long)

    data_g = Data(
        x=x_g.cpu(),
        edge_index=edge_index_g_local.cpu()
    )
    G = to_networkx(data_g, to_undirected=True)

    # 节点布局
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # ========= 6. 开始画图 =========
    plt.figure(figsize=(6, 6))

    # 6.1 画节点：FS 节点更红
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_vals,
        cmap=plt.cm.Reds,
        edgecolors="black",
        node_size=300
    )

    # 6.2 画原始边：根据 p_keep 分成保留 / 删除（颜色统一）
    if len(orig_edges) > 0:
        orig_scores_np = np.array(orig_scores)

        solid_edges = []
        dashed_edges = []

        for edge, score in zip(orig_edges, orig_scores_np):
            if score >= tau_keep:
                solid_edges.append(edge)  # 保留：实线
            else:
                dashed_edges.append(edge)  # 删除：虚线

        # 保留的边：统一蓝色实线
        if solid_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=solid_edges,
                edge_color="blue",
                width=2.0,
            )

        # 倾向删除的边：统一蓝色虚线
        if dashed_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=dashed_edges,
                edge_color="blue",
                width=2.0,
                style="dashed",
            )

    # 6.3 画候选新增边：红色虚线，p_add >= tau_add
    if len(new_edges) > 0:
        new_scores_np = np.array(new_scores)
        # 只画分数高于 tau_add 的
        filtered_edges = []
        filtered_scores = []
        for e, s in zip(new_edges, new_scores_np):
            if s >= tau_add:
                filtered_edges.append(e)
                filtered_scores.append(s)

        if filtered_edges:
            filtered_scores_np = np.array(filtered_scores)
            norm_add = plt.Normalize(vmin=filtered_scores_np.min(), vmax=filtered_scores_np.max())
            new_colors_rgba = plt.cm.Reds(norm_add(filtered_scores_np))

            nx.draw_networkx_edges(
                G, pos,
                edgelist=filtered_edges,
                edge_color=new_colors_rgba,
                width=2.0,
                style="dashed",
            )

    # 6.4 标签 & colorbar
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 节点 FS 程度的 colorbar
    cb_nodes = plt.colorbar(nodes, shrink=0.6, pad=0.02)
    cb_nodes.set_label("FS node score (bool/mask)", fontsize=10)

    # 原始边 p_keep 的 colorbar
    if len(orig_edges) > 0:
        sm_keep = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
        sm_keep.set_array([])
        cb_keep = plt.colorbar(sm_keep, shrink=0.6, pad=0.02)
        cb_keep.set_label("p_keep (edge retention prob.)", fontsize=10)

    # 标题：简单用 y_desired 展示（如果你有 y_hat，可以自己改）
    if y_desired is not None:
        y_des_val = int(y_desired[g_idx].item()) if y_desired.dim() > 0 else int(y_desired.item())
        plt.title(f"Graph #{g_idx} CF explanation (y_desired={y_des_val})")
    else:
        plt.title(f"Graph #{g_idx} CF explanation")

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