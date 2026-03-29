import os
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical


def subgraph_mining(args,datasets):
    if args.subgraph_method == 'genGraphEx':
        patterns_0 = []
        patterns_1 = []
        if args.dataset == 'ba2motif':
            Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[0],25)
            for i in range(50):
                patterns_0.append(graphsampler(25,Bdist, mean_estimate, result, Adj))

            Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[1],25)
            for i in range(50):
                patterns_1.append(graphsampler(25,Bdist, mean_estimate, result, Adj))
        # mutag 417, nci1 111
        if args.dataset == 'mutag':
            X, Adj = GraphRepModelDiscrete(datasets[0],417)
            for i in range(100):
                patterns_0.append(graphsamplerDiscrete(417,X, Adj))
            X, Adj = GraphRepModelDiscrete(datasets[1],417)
            for i in range(100):
                patterns_1.append(graphsamplerDiscrete(417,X, Adj))
        if args.dataset == 'nci1':
            X, Adj = GraphRepModelDiscrete(datasets[0],111)
            for i in range(100):
                patterns_0.append(graphsamplerDiscrete(111,X, Adj))
            X, Adj = GraphRepModelDiscrete(datasets[1],111)
            for i in range(100):
                patterns_1.append(graphsamplerDiscrete(111,X, Adj))
        if args.dataset == 'proteins':
            # 槽位行数 N 须 ≥ 该类内最大节点数，否则 X[j] 行越界（PROTEINS 常有图 >75 节点）
            nf0 = max((int(d.x.shape[1]) for d in datasets[0]), default=1)
            nf1 = max((int(d.x.shape[1]) for d in datasets[1]), default=1)
            nf = max(nf0, nf1)
            N0 = max((int(d.num_nodes) for d in datasets[0]), default=1)
            N1 = max((int(d.num_nodes) for d in datasets[1]), default=1)
            X, Adj = GraphRepModelDiscrete(datasets[0], N0)
            for i in range(100):
                patterns_0.append(graphsamplerDiscrete(N0, X, Adj, num_node_features=nf))
            X, Adj = GraphRepModelDiscrete(datasets[1], N1)
            for i in range(100):
                patterns_1.append(graphsamplerDiscrete(N1, X, Adj, num_node_features=nf))
        # 对patterns中的所有图按照key1(节点数)和key2(度数)进行排序
        sort_key = lambda G: (G.number_of_nodes(), nx.density(G))
        patterns_0.sort(key=sort_key, reverse=True)
        patterns_1.sort(key=sort_key, reverse=True)
        return {0: patterns_0, 1: patterns_1}

# # 连续节点特征
def GraphRepModel(classdata,N):
    """

    :param N:
    :param classdata:
    :return: Bdist: 节点存在的概率分布
             mean_estimate: 节点特征的均值向量
             result: 节点特征的协方差矩阵
             Adj: 边的存在概率矩阵
    """
    B=np.zeros(N)# Node Type Matrix for nodes of 10 types
    X=np.zeros((N,10))
    numgraphs=len(classdata)
    workingdata=classdata

    #Learning the on/off bit and node representations
    for i in range(numgraphs):#len(data1))
        data=workingdata[i]
        x=data.x
        k=len(data.x) # keeping tab of the number of nodes in the ith graph
        #print(k)
        x=x.numpy()

        B[0:k]+=1
        #print(B)
        X[:x.shape[0], :] += x
    B1=B
    B=np.reshape(B,(N,-1))
    mean_estimate=X/B

    #print(mean_estimate)
    covarr=np.zeros((10,10,N))

    for i in range(numgraphs):
        data=workingdata[i]
        x=data.x
        #print(len(x))
        x=x.numpy()
        subtracted_array = x-mean_estimate[:x.shape[0], :]
        result_matrices=[]
        for row in subtracted_array:
          element = row.reshape(10, 1)  # Reshape to a 2x1 element
          result_matrix = np.dot(element, element.T)  # Multiply by its transpose
          result_matrices.append(result_matrix)

    # Concatenate the resulting 2x2 matrices along the third dimension to create a 3D array
        result_array = np.stack(result_matrices, axis=2)
        #print(result_array.shape)
        covarr[:, :, :result_array.shape[2]] += result_array
    print(covarr.shape)
    covariance_estimate=covarr/B[:,None,None]
    #print(covariance_estimate.shape)
    print(covariance_estimate.shape)
    result_list=[]
    for i in range(N):
        result_list.append(covarr[:, :, i] / B[i])

    # Convert the list of results back to a NumPy array
    result = np.stack(result_list, axis=2)
    print(result.shape)
    Bdist=B/numgraphs
    Adj=np.zeros((N,N))# Edge type count for only two types edge present/edge absent
    for i in range(len(workingdata)):
        data=workingdata[i]
        adj=data.edge_index
        rowlen=len(adj[0][:])
        #print(rowlen)
        #print(adj[:][0])
        #print(adj[:][1])
        for j in range(rowlen):
            k1=adj[0][j]
            k2=adj[1][j]
            Adj[k1][k2]+=1

    #Learning the parameters for the distribution of nodes
    #numgraphs=len(data1)
    #X=X/numgraphs #converting X to the node distribution matrix
    Adj=Adj/numgraphs
    return Bdist, mean_estimate, result, Adj

def graphsampler(N, Bdist, mean_estimate, result, Adj, threshold=0.97, visualize=False):
    """
    Gen-GraphEx 生成器（集成阈值过滤版）

    Args:
        N (int): 最大节点数
        Bdist: 节点存在概率分布
        mean_estimate: 节点特征均值
        result: 节点特征协方差
        Adj: 边存在概率矩阵
        threshold (float): 边概率阈值 (0.0~1.0). 低于此值的边将不会被生成.
                           推荐值: 0.8 或 0.9 用于提取 Motif, 0.0 用于保留所有细节.

    Returns:
        nx.Graph: 生成的 NetworkX 图对象
    """

    # --- 1. 采样节点存在性 (Node Existence) ---
    # Bdist 是一个列表的列表，需要展平
    Bdist_flat = np.concatenate(Bdist)
    # 伯努利采样：决定每个索引位置的节点是否存在
    samples = [np.random.choice([0, 1], p=[1 - p, p]) for p in Bdist_flat]

    # --- 2. 采样节点特征 (Node Features) ---
    nodefeat_list = []
    for i in range(N):
        if samples[i] != 0:  # 如果该位置节点被采样存在
            meanvec = mean_estimate[i, :]
            covarvec = result[:, :, i]
            # 多元高斯分布采样特征
            # 注意：covarvec 可能需要处理数值稳定性
            normalsamples = np.random.multivariate_normal(meanvec, covarvec, 1)
            nodefeat_list.append(normalsamples)

    # 如果没有采样到任何节点，返回空图 (防止报错)
    if not nodefeat_list:
        return nx.Graph()

    # 堆叠特征矩阵
    nodefeat = np.vstack(nodefeat_list)  # Shape: [k, feature_dim]
    k = nodefeat.shape[0]  # k 为实际生成的节点数量

    # --- 3. 采样邻接矩阵 (Adjacency Matrix) ---
    # [应用方法一]：概率阈值过滤
    # 复制 Adj 以免修改外部传入的原始数据
    Adj_filtered = Adj.copy()

    # 核心逻辑：将低于阈值的边连接概率直接置为 0
    # 这意味着这些边在下面的 binomial 采样中不可能被生成 (p=0)
    Adj_filtered[Adj_filtered < threshold] = 0

    # 伯努利采样：基于过滤后的概率矩阵决定边是否存在
    Adjacencymat = np.random.binomial(1, Adj_filtered)

    # --- 4. 截断与对齐 (Truncation) ---
    # 根据实际存在的节点数 k，将多余的行列置零
    if k < N:
        Adjacencymat[k:, :] = 0
        Adjacencymat[:, k:] = 0

    # 取出有效的 k*k 子矩阵
    Adjacencymat_k = Adjacencymat[:k, :k]

    # --- 5. 构建 NetworkX 图 ---
    # 将邻接矩阵转换为 NetworkX 图
    G = nx.from_numpy_array(Adjacencymat_k, create_using=nx.Graph())

    # (可选) 将生成的节点特征作为属性赋予节点
    for i in range(k):
        # 注意 nodefeat[i] 是一个数组，可能需要 flatten
        G.nodes[i]['x'] = nodefeat[i].flatten()

    # --- [新增功能] 6. 仅保留最大连通子图 ---
    # 获取所有连通分量，按节点数降序排列
    largest_cc_nodes = max(nx.connected_components(G), key=len)

    # 提取子图 (.copy() 切断引用，防止后续修改影响原图)
    G = G.subgraph(largest_cc_nodes).copy()

    # 重置节点索引为 0, 1, 2... (保持特征属性跟随节点移动)
    # 这对于后续如果还要将图转回矩阵或输入 GNN 很重要
    # G = nx.convert_node_labels_to_integers(G)

    # --- 绘图 (可选) ---
    if visualize:
        # 使用 spring_layout 布局，seed 固定以保证结果可复现
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos=pos, node_size=50, node_color='red',
                         edge_color='gray', with_labels=True, width=1.5)
        plt.title(f"Generated Graph (Threshold={threshold})")
        plt.show()
        plt.close()

    return G


# 离散节点特征

def GraphRepModelDiscrete(targetclass, N=111):
    """
    Gen-GraphEx 图表示模型（通用适配版）
    自动适配 targetclass 中的特征维度。
    """
    if len(targetclass) == 0:
        return np.zeros((N, 1)), np.zeros((N, N))

    # 列：one-hot 维数取子集最大值（与同集内维数一致时与「只看首图」等价）
    feat_dim = max(int(d.x.shape[1]) for d in targetclass)
    num_cols = feat_dim + 1  # 第 0 列为空槽位，1..feat_dim 为类别

    X = np.zeros((N, num_cols))
    Adj = np.zeros((N, N))

    print(f"Detected feat_dim={feat_dim}, slot_rows N={N}. X matrix shape: {X.shape}")
    print(f"Processing {len(targetclass)} graphs...")

    # --- 统计节点分布 ---
    for i in range(len(targetclass)):
        data = targetclass[i]

        if data.x.shape[1] > 1:
            x_indices = torch.argmax(data.x, dim=1)
        else:
            x_indices = data.x.squeeze()

        x_indices = x_indices + 1  # 映射到列 1..feat_dim

        current_num_nodes = len(x_indices)
        # 行 j 表示「第 j 个槽位」：仅统计前 N 个节点，避免图节点数 > N 时 X[j] 行越界
        slot_nodes = min(current_num_nodes, N)
        for j in range(slot_nodes):
            type_idx = int(x_indices[j].item())
            if 0 <= type_idx < num_cols:
                X[j][type_idx] += 1

        # 槽位 slot_nodes..N-1 视为「该图在此位置无节点」
        for k in range(slot_nodes, N):
            X[k][0] += 1

    # --- 统计边分布 ---
    for i in range(len(targetclass)):
        data = targetclass[i]
        adj = data.edge_index
        if adj.shape[1] > 0:
            for j in range(adj.shape[1]):
                k1 = adj[0][j].item()
                k2 = adj[1][j].item()
                if k1 < N and k2 < N:
                    Adj[k1][k2] += 1

    # --- 归一化 ---
    numgraphs = len(targetclass)
    if numgraphs > 0:
        X = X / numgraphs
        Adj = Adj / numgraphs

    return X, Adj


def graphsamplerDiscrete(N, X, Adj, threshold=0.3, num_node_features=37, visualize=False):
    """
    Gen-GraphEx 生成器（通用适配版）

    Args:
        num_node_features (int): 数据集的原始特征维度 (例如 37)
                                 必须与训练 GNN 时的维度一致！
    """

    # --- 1. 采样节点类型 ---
    node_types_list = []
    valid_indices = []

    for i in range(N):
        probs = torch.from_numpy(X[i][:])
        m = Categorical(probs)
        chosen_type = m.sample().item()

        if chosen_type != 0:
            node_types_list.append(chosen_type)
            valid_indices.append(i)

    if not valid_indices:
        return nx.Graph()

    # --- 2. 采样邻接矩阵 (保持不变) ---
    Adj_filtered = Adj.copy()
    if threshold > 0:
        Adj_filtered[Adj_filtered < threshold] = 0

    Adj_tensor = torch.from_numpy(Adj_filtered)
    Adjacency = torch.bernoulli(Adj_tensor)
    Adjacency = torch.tril(Adjacency) + torch.tril(Adjacency, -1).t()

    # --- 3. 提取子图 ---
    valid_idx_tensor = torch.tensor(valid_indices).long()
    sub_adj = Adjacency[valid_idx_tensor][:, valid_idx_tensor]

    # --- 4. 构建图 ---
    adj_np = sub_adj.numpy()
    G = nx.from_numpy_array(adj_np, create_using=nx.Graph())

    # --- 5. 设置节点特征 ---
    node_labels = {}

    for i, type_idx in enumerate(node_types_list):
        # 还原特征索引: 1~37 -> 0~36
        feat_idx = type_idx - 1

        # 动态创建 One-Hot 向量 (长度为 num_node_features)
        one_hot = np.zeros(num_node_features)

        # 安全赋值
        if 0 <= feat_idx < num_node_features:
            one_hot[feat_idx] = 1.0
            # 既然没有原子映射表，直接显示类型ID
            label_text = f"{feat_idx}"
        else:
            label_text = "?"

        G.nodes[i]['type'] = type_idx
        G.nodes[i]['x'] = one_hot
        G.nodes[i]['label_name'] = label_text  # 存入临时属性

    # 保留最大连通子图
    if G.number_of_nodes() > 0:
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()
        G = nx.convert_node_labels_to_integers(G)

    # --- 6. 可视化 ---
    if visualize:
        node_labels = {i: G.nodes[i].get('label_name', '?') for i in G.nodes()}
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42, k=0.5)

        nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6', edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='sans-serif')

        plt.title(f"Generated Graph (Threshold={threshold})", fontsize=15)
        plt.axis('off')
        plt.show()
        plt.close()

    return G