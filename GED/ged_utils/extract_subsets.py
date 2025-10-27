import torch
from torch_geometric.data import DataLoader
from gnns import mutag_gnn # 假设Mutag_GCN定义在gnns模块中

def extract_class_subsets(model, train_dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    从训练集中提取被GCN分类为0类和1类的子集

    参数:
        model: 训练好的GCN模型
        train_dataset: 训练数据集 (PyTorch Geometric DataLoader or Dataset)
        device: 计算设备 (cuda或cpu)

    返回:
        class_0_indices: 分类为0类的样本索引列表
        class_1_indices: 分类为1类的样本索引列表
    """
    model.eval()  # 设置模型为评估模式
    model.to(device)

    class_0_indices = []
    class_1_indices = []

    # 如果train_dataset是PyTorch Geometric的Dataset，转换为DataLoader
    if not isinstance(train_dataset, DataLoader):
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    else:
        train_loader = train_dataset

    with torch.no_grad():  # 禁用梯度计算
        for i, data in enumerate(train_loader):
            data = data.to(device)
            out = model(data)  # 前向传播
            pred = out.argmax(dim=1)  # 获取预测类别

            if pred.item() == 0:
                class_0_indices.append(i)
            elif pred.item() == 1:
                class_1_indices.append(i)

    return class_0_indices, class_1_indices


# 示例用法
def main():
    # 假设你已经有了训练好的GCN模型和数据集
    # model = Mutag_GCN()
    gnn_path = "../../param/gnns/mutag_gcn.pt"
    # train_dataset = YourDataset()
    loaded_data = torch.load(gnn_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = torch.load(gnn_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #
    # # 检查加载的内容是模型还是状态字典
    # if isinstance(loaded_data, torch.nn.Module):
    #     # 如果是整个模型对象
    #     model = loaded_data
    # else:
    #     # 如果是状态字典，需先定义模型架构
    #     model.load_state_dict(loaded_data['model_state_dict'])  # 加载状态字典
    model.eval()  # 设置为测试模式
    # 获取分类子集的索引
    class_0_indices, class_1_indices = extract_class_subsets(model, train_dataset)

    # 打印结果
    print(f"Class 0 samples: {len(class_0_indices)} indices: {class_0_indices}")
    print(f"Class 1 samples: {len(class_1_indices)} indices: {class_1_indices}")

    # 可选：提取子集数据
    class_0_subset = [train_dataset[i] for i in class_0_indices]
    class_1_subset = [train_dataset[i] for i in class_1_indices]

    return class_0_subset, class_1_subset


if __name__ == "__main__":
    main()