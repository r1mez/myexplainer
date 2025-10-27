"""
检查MUTAG数据集的最大节点数
"""
import sys
sys.path.insert(0, '.')

from utils import get_datasets

# 加载数据集
print("Loading mutag dataset...")
train_dataset, val_dataset, test_dataset = get_datasets(name='mutag')

# 检查各个数据集
all_datasets = {
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
}

print("\nDataset statistics:")
print("=" * 60)

for name, dataset in all_datasets.items():
    num_nodes_list = [data.num_nodes for data in dataset]
    max_nodes = max(num_nodes_list)
    min_nodes = min(num_nodes_list)
    avg_nodes = sum(num_nodes_list) / len(num_nodes_list)

    print(f"\n{name.upper()} set:")
    print(f"  Size: {len(dataset)}")
    print(f"  Min nodes: {min_nodes}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Avg nodes: {avg_nodes:.2f}")

# 检查整体
all_data = list(train_dataset) + list(val_dataset) + list(test_dataset)
all_num_nodes = [data.num_nodes for data in all_data]
overall_max = max(all_num_nodes)
overall_min = min(all_num_nodes)
overall_avg = sum(all_num_nodes) / len(all_num_nodes)

print("\n" + "=" * 60)
print("OVERALL statistics:")
print(f"  Total graphs: {len(all_data)}")
print(f"  Min nodes: {overall_min}")
print(f"  Max nodes: {overall_max}")
print(f"  Avg nodes: {overall_avg:.2f}")
print("=" * 60)

print(f"\nRecommended max_num_nodes: {overall_max}")
print(f"With safety margin: {overall_max + 2}")
