def visualize_subgraph(self, data, edge_mask, k=10):
    """
    可视化GNN解释子图，突出显示前k条边，节点按特征类型用不同颜色区分。

    参数：
        data (Data): PyG的Data对象，包含edge_index、x
        edge_mask (torch.Tensor): 边重要性掩码，形状[num_edges]
        k (int): 选择前k条边，默认为15
    """
    # 确保掩码为一维
    # node_mask = node_mask.squeeze()
    edge_mask = edge_mask.squeeze()

    # 获取前k条边（值最大的边）
    k = min(k, edge_mask.shape[0])  # 避免k超过边数
    _, selected_edge_indices = torch.topk(edge_mask, k=k, largest=True)
    masked_edge_index = data.edge_index[:, selected_edge_indices]

    # 创建NetworkX图
    G = nx.Graph()
    num_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0

    G.add_nodes_from(range(num_nodes))
    edge_list = data.edge_index.t().cpu().numpy().tolist()
    G.add_edges_from(edge_list)

    # 创建子图（只包含前k条边）
    G_sub = nx.Graph()
    sub_edge_list = masked_edge_index.t().cpu().numpy().tolist()
    G_sub.add_edges_from(sub_edge_list)

    # 节点颜色：基于data.x（节点特征）通过KMeans聚类区分类型
    node_colors = ['lightblue' for _ in range(num_nodes)]  # 默认颜色
    if hasattr(data, 'x') and data.x is not None and data.x.shape[0] > 0:
        node_features = data.x.cpu().numpy()
        # 检查特征是否有效（无NaN、无Inf、节点数足够）
        if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
            print("Warning: node_features contains NaN or Inf. Using default colors.")
        elif num_nodes < 2:
            print("Warning: Too few nodes ({}) for clustering. Using default colors.".format(num_nodes))
        else:
            # 动态设置簇数（不超过节点数）
            n_clusters = min(3, num_nodes)
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                kmeans.fit(node_features)  # 确保拟合
                node_labels = kmeans.labels_  # 获取聚类标签
                unique_labels = set(node_labels)
                color_map = plt.cm.get_cmap('Set1', max(len(unique_labels), 1))  # 避免0簇
                node_colors = [color_map(node_labels[i]) for i in range(num_nodes)]
                print(f"KMeans successful: {n_clusters} clusters, labels: {node_labels}")
            except Exception as e:
                print(f"KMeans failed: {e}. Using default colors.")
    else:
        print("No valid node features (data.x). Using default colors.")

    # 绘制图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # 固定种子以确保布局可重现

    # 绘制原图（淡色）
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.3)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 绘制子图（只高亮边）
    nx.draw_networkx_edges(G_sub, pos, edge_color='red', width=2.0)

    plt.title("GNN Explanation: Original Graph (gray) and Important Edges (red)")
    plt.show()