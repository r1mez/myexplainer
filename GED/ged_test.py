import torch
import torch_geometric as tg
from neurosed import models

# 1. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.NormGEDModel(
    n_layers=3,           # 图卷积层数
    input_dim=16,         # 输入特征维度
    hidden_dim=64,        # 隐藏层维度
    output_dim=32,        # 输出嵌入维度
    conv='gin',           # 图卷积类型
    pool='add',           # 全局池化方法
    device=device         # 运行设备
).to(device)

# 2. 准备数据（假设有两个图数据）
graph1 = tg.data.Data(
    x=torch.randn(10, 16),    # 10个节点，每个节点16维特征
    edge_index=tg.utils.erdos_renyi_graph(10, 0.3)  # 随机生成边
)

graph2 = tg.data.Data(
    x=torch.randn(10, 16),     # 8个节点，每个节点16维特征
    edge_index=tg.utils.erdos_renyi_graph(10, 0.4)   # 随机生成边
)

# 3. 直接计算两个图的归一化GED
batch1 = tg.data.Batch.from_data_list([graph1]).to(device)
batch2 = tg.data.Batch.from_data_list([graph2]).to(device)

with torch.no_grad():
    norm_ged = model(batch1, batch2)
    print(f"预测的归一化GED: {norm_ged.item():.4f}")

# 4. 批量计算多个图对之间的距离
graphs = [graph1, graph2]
batch = tg.data.Batch.from_data_list(graphs).to(device)

with torch.no_grad():
    # 计算所有图对之间的距离矩阵
    distance_matrix = model.predict_outer(graphs, graphs)
    print("距离矩阵形状:", distance_matrix.shape)

# 5. 使用预计算的目标嵌入加速查询
# 假设graph2是目标图，graph1是查询图
model.embed_targets([graph2])  # 预计算目标图的嵌入

with torch.no_grad():
    # 快速计算查询图与预计算目标的距离
    distances = model.predict_outer_with_queries([graph1])
    print(f"快速计算的归一化GED: {distances.item():.4f}")