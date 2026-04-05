import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import torch.nn as nn
from gnns import *
from torch_geometric.data import InMemoryDataset
from utils import get_datasets
from utils.baseline_eval_metrics import (
    compute_proximity_from_edge_index,
    compute_fidelity_prob_from_probs,
    compute_sparsity_from_edge_index,
    OracleWrappedModel,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def vector_to_symm_matrix(
	vector: torch.Tensor, n_rows: int, n_rows_pad: int = 0, offset:int = -1
) -> torch.Tensor:
	"""
	Converts a vector into a matrix by using values of the vector to fill in the lower
	triangle of the matrix and adds the transpose to create a symmetric matrix. Add n_rows_pad
	to create a matrix of shape (n_rows + n_rows_pad, n_rows + n_rows_pad) where the padding
	is added to the ends of the matrix (bottom and right sides)\n
	See example in symm_matrix_to_vector
	"""
	matrix = torch.zeros(n_rows, n_rows).to(vector.device)
	idx = torch.triu_indices(n_rows, n_rows, offset=offset)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.triu(matrix) + torch.triu(matrix, -1).t()
	return F.pad(symm_matrix, (0, n_rows_pad, 0, n_rows_pad))

class CFExplainer(nn.Module):
	def __init__(self, pred_model: nn.Module, device: str, lr: float) -> None:
		super().__init__()
		self.pred_model = pred_model
		self.device = device
		self.lr = lr

		self.init_parameters()

	def init_parameters(self) -> None:
		"""
		冻结预测模型的参数
		"""
		for name, param in self.pred_model.named_parameters():
			if name.endswith("weight") or name.endswith("bias"):
				param.requires_grad = False

	def get_P_vec(self, dim: int) -> Parameter:
		"""
		P_vec 是 explainer 针对每个图要学习的参数（对称矩阵的上三角展平）
		"""
		cur_P_vec_size = int((dim * dim + dim) / 2)
		eps = 10**-4
		with torch.no_grad():
			cur_P_vec = Parameter(
				torch.sub(torch.randn(cur_P_vec_size, device=self.device), eps),
				requires_grad=True,
			)
		return cur_P_vec

	def forward(
		self,
		x: torch.Tensor,              # [N, F]
		edge_index: torch.Tensor,     # [2, E]
		batch: torch.Tensor,          # [N]
		adj: torch.Tensor,            # [N, N]
		p_vec: Parameter
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		使用 p_vec 生成软/离散的边权重，并通过 GNN 的 get_pred_explain 得到预测。

		返回:
			pred_output: 使用软 edge_mask 的 logits（可导）
			actual_pred_output: 使用离散 edge_mask 的 logits
			cf_adj: 离散化后的邻接矩阵 (0/1)
		"""
		num_nodes = adj.size(0)

		# 1. 用 p_vec 生成对称矩阵
		p_hat_symm_matrix = vector_to_symm_matrix(p_vec, num_nodes, offset=0)  # [N, N]

		# 和原始 adj 做 mask，只在原有边上调整权重
		A_tilde = torch.sigmoid(p_hat_symm_matrix) * adj  # [N, N]
		A_tilde.requires_grad_()

		# 2. 从 A_tilde 中抽取每条边的权重，得到 edge_mask_soft
		row, col = edge_index  # [E], [E]
		edge_mask_soft = A_tilde[row, col]  # [E]

		# 用 get_pred_explain，让 GNN 使用 edge_weight = edge_mask_soft
		probs_soft, logits_soft = self.pred_model.get_pred_explain(
			x, edge_index, edge_mask_soft, batch
		)

		# 3. 阈值化，得到离散 cf_adj 和对应 edge_mask_cf
		threshold_p_hat = (torch.sigmoid(p_hat_symm_matrix) >= 0.5).float()
		cf_adj = threshold_p_hat * adj  # [N, N]
		cf_adj.requires_grad_()

		edge_mask_cf = cf_adj[row, col]  # [E]，0/1

		probs_cf, logits_cf = self.pred_model.get_pred_explain(
			x, edge_index, edge_mask_cf, batch
		)

		# 这里返回的是 logits，后面在 loss 里再做 log_softmax
		return logits_soft, logits_cf, cf_adj

	def loss(
		self,
		log_probs: torch.Tensor,   # [1, num_classes]
		label: torch.Tensor,       # 标量，原始标签
		pred: torch.Tensor,        # 标量，counterfactual 下预测类别
		cf_adj: torch.Tensor,      # [N, N]
		adj: torch.Tensor          # [N, N]
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

		if label.dim() > 0:
			label_scalar = label.item()
		else:
			label_scalar = int(label)

		# 预测损失：希望远离原类
		target = torch.tensor([label_scalar], device=self.device)
		loss_pred = -F.nll_loss(log_probs, target)

		loss_graph_dist = torch.sum(torch.abs(cf_adj - adj)) / 2

		label_prediction_matches = 1 if (label_scalar == int(pred)) else 0
		loss_total = label_prediction_matches * loss_pred * 1e5 + loss_graph_dist
		return loss_total, loss_pred, loss_graph_dist

	def run_one_graph(
			self,
			x: torch.Tensor,
			edge_index: torch.Tensor,
			batch: torch.Tensor,
			adj: torch.Tensor,
			label: torch.Tensor,
			epochs: int
	) -> torch.Tensor:

		# 初始就用原图邻接矩阵作为一个“候选 cf”
		best_cf_adj = adj.detach().clone()
		best_loss_total = float("inf")

		p_vec = self.get_P_vec(adj.shape[0])
		optimizer = optim.SGD([p_vec], lr=self.lr)

		for _ in range(epochs):
			optimizer.zero_grad()

			logits_soft, logits_cf, cf_adj = self(x, edge_index, batch, adj, p_vec)

			y_pred_actual = torch.argmax(logits_cf, dim=1)[0]
			log_probs = F.log_softmax(logits_soft, dim=1)

			loss_total, _, _ = self.loss(
				log_probs, label, y_pred_actual, cf_adj, adj
			)

			loss_total.backward()
			optimizer.step()

			# 不再要求 y_pred_actual != label，只要 loss 更小就更新
			if loss_total.item() < best_loss_total:
				best_cf_adj = cf_adj.detach().clone()
				best_loss_total = loss_total.item()

		return best_cf_adj

def run_cf_gnnexplainer(
	pred_model: nn.Module, pred_labels, epochs: int, device: str, lr: float, dataset: InMemoryDataset
):
	"""
	运行CF-GNNExplainer，为数据集中的每个图生成反事实解释

	返回：
		cf_feat: 节点特征列表
		cf_adj: 反事实邻接矩阵列表（稠密格式）
		cf_edge: 边特征列表（CFGNNExplainer不修改边特征，返回None）
		graph_idx: 成功生成CF的图索引
	"""
	pred_model.eval()
	explainer = CFExplainer(pred_model, device=device, lr=lr)
	cf_feat, cf_adj, cf_edge, graph_idx = [], [], [], []

	for idx in tqdm(range(len(dataset))):
		data = dataset[idx]
		x = data.x.to(device)
		edge_index = data.edge_index.to(device)
		batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

		# 稠密邻接矩阵 [N, N]
		adj = to_dense_adj(edge_index).squeeze(0).to(device)

		y_pred = pred_labels[idx].to(device)  # label（这里是 GNN 在原图上的预测）

		best_cf_adj = explainer.run_one_graph(x, edge_index, batch, adj, y_pred, epochs)

		if best_cf_adj.shape[0] != 0:
			cf_feat.append(x)
			cf_adj.append(best_cf_adj)
			# 这里暂时没有专门的 edge_attr / edge_mask 结构，就先占位
			cf_edge.append(None)
			graph_idx.append(torch.tensor(idx, device=device))

	# 注意：val_dataset 可能没有 train_idx，这里可以改成 len(dataset)
	print(f"Dataset completed! {len(cf_feat)}/{len(dataset)} cfs found")
	return cf_feat, cf_adj, cf_edge, graph_idx


# def evaluate_cf_gnnexplainer(pred_model, dataset, device, lr=0.01, epochs=100):
# 	"""
# 	评估CF-GNNExplainer，使用与MyExplainerV2相同的评估指标
#
# 	参数：
# 		pred_model: 预训练的GNN分类器
# 		dataset: 验证/测试数据集
# 		device: 计算设备
# 		lr: CFGNNExplainer的学习率
# 		epochs: 每个图的优化轮数
#
# 	返回：
# 		metrics: 包含validity, proximity, fidelity_prob, sparsity的字典
# 	"""
# 	from torch_geometric.data import Batch, Data
# 	from torch_geometric.utils import dense_to_sparse
#
# 	print("\n" + "="*60)
# 	print("Evaluating CF-GNNExplainer")
# 	print("="*60)
#
# 	pred_model.eval()
# 	explainer = CFExplainer(pred_model, device=device, lr=lr)
#
# 	# 预计算所有图的原始预测和目标标签
# 	y_desired_list = []
# 	ori_graphs_list = []
# 	ori_prob_list = []
#
# 	print("\n1. Computing original predictions...")
# 	with torch.no_grad():
# 		for data in tqdm(dataset):
# 			data = data.to(device)
# 			ori_pred_logits = pred_model(data.x, data.edge_index, data.batch)
# 			ori_prob = F.softmax(ori_pred_logits, dim=1)[0]  # [num_classes]
# 			ori_pred = ori_pred_logits.argmax(dim=1).item()
# 			y_desired = 1 - ori_pred  # 反事实标签
# 			y_desired_list.append(y_desired)
# 			ori_graphs_list.append(data)
# 			ori_prob_list.append(ori_prob)
#
# 	# 评估指标
# 	valid_cf = 0
# 	proximity_sum = 0.0
# 	fidelity_prob_sum = 0.0  # 概率下降总和
# 	sparsity_sum = 0.0
# 	total_graphs = 0
# 	successful_cf = 0
#
# 	print("\n2. Generating counterfactuals and computing metrics...")
# 	for idx in tqdm(range(len(dataset))):
# 		total_graphs += 1
# 		ori_data = ori_graphs_list[idx]
# 		x = ori_data.x
# 		edge_index = ori_data.edge_index
# 		batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
# 		y_desired = y_desired_list[idx]
# 		ori_prob = ori_prob_list[idx]
# 		ori_pred = 1 - y_desired  # 原始预测类
#
# 		# 原图的稠密邻接矩阵
# 		ori_adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0).to(device)
#
# 		# 生成反事实邻接矩阵
# 		y_pred_tensor = torch.tensor([ori_pred], device=device)  # 原始预测
# 		cf_adj = explainer.run_one_graph(x, edge_index, batch, ori_adj, y_pred_tensor, epochs)
#
# 		# 将cf_adj转换为PyG Data格式
# 		cf_edge_index, _ = dense_to_sparse(cf_adj)
# 		cf_data = Data(
# 			x=x.cpu(),
# 			edge_index=cf_edge_index.cpu(),
# 			num_nodes=x.size(0)
# 		)
#
# 		# 1. Validity：检查CF图是否预测为目标类
# 		with torch.no_grad():
# 			cf_data_dev = cf_data.to(device)
# 			cf_pred_logits = pred_model(
# 				cf_data_dev.x,
# 				cf_data_dev.edge_index,
# 				torch.zeros(cf_data_dev.x.size(0), dtype=torch.long, device=device)
# 			)
# 			cf_prob = F.softmax(cf_pred_logits, dim=1)[0]  # [num_classes]
# 			cf_pred = cf_pred_logits.argmax(dim=1).item()
#
# 		if cf_pred == y_desired:
# 			valid_cf += 1
# 			successful_cf += 1  # 这里表示“真正翻到目标类”的个数
#
# 		# 2. Proximity：计算原图和CF图的邻接矩阵距离
# 		ori_adj_np = ori_adj.cpu().numpy()
# 		cf_adj_np = cf_adj.cpu().numpy()
#
# 		# Frobenius范数
# 		adj_diff = np.linalg.norm(ori_adj_np - cf_adj_np, ord='fro')
#
# 		# 归一化：用边数
# 		m_ori = ori_data.num_edges // 2  # 无向图
# 		m_cf = cf_data.num_edges // 2
# 		max_m = max(m_ori, m_cf, 1)
#
# 		proximity = adj_diff / max_m
# 		proximity_sum += proximity
#
# 		# 3. Fidelity (Probability)：原图对原始类的概率 - 反事实图对原始类的概率
# 		ori_prob_on_ori_class = ori_prob[ori_pred].item()
# 		cf_prob_on_ori_class = cf_prob[ori_pred].item()
# 		fidelity_prob = ori_prob_on_ori_class - cf_prob_on_ori_class  # 概率下降量
# 		fidelity_prob_sum += fidelity_prob
#
# 		# 4. Sparsity：计算原图和CF图的边集差异
# 		ori_edge_set = set()
# 		for i in range(edge_index.size(1)):
# 			u, v = edge_index[0, i].item(), edge_index[1, i].item()
# 			ori_edge_set.add((min(u, v), max(u, v)))
#
# 		cf_edge_set = set()
# 		for i in range(cf_edge_index.size(1)):
# 			u, v = cf_edge_index[0, i].item(), cf_edge_index[1, i].item()
# 			cf_edge_set.add((min(u, v), max(u, v)))
#
# 		# 解释边 = 变化的边（原图有但CF没有 + CF有但原图没有）
# 		exp_edges = ori_edge_set.symmetric_difference(cf_edge_set)
#
# 		num_exp_edges = len(exp_edges)
# 		num_ori_edges = len(ori_edge_set)
# 		sparsity = 1 - (num_exp_edges / max(num_ori_edges, 1))
# 		sparsity_sum += sparsity
#
# 	# 计算平均指标
# 	validity = valid_cf / total_graphs  # 仍然只看翻到目标类的比例
# 	avg_proximity = proximity_sum / total_graphs
# 	avg_fidelity_prob = fidelity_prob_sum / total_graphs
# 	avg_sparsity = sparsity_sum / total_graphs
#
#
# 	print("\n" + "="*60)
# 	print("Evaluation Results:")
# 	print("="*60)
# 	print(f"  Validity ↑: {validity:.4f} (successful: {valid_cf}/{total_graphs})")
# 	print(f"  Proximity ↓: {avg_proximity:.4f}")
# 	print(f"  Fidelity (Prob Drop) ↑: {avg_fidelity_prob:.4f}")
# 	print(f"  Sparsity ↑: {avg_sparsity:.4f}")
# 	print(f"  CF Generation Rate: {successful_cf}/{total_graphs} ({successful_cf/max(total_graphs,1):.2%})")
# 	print("="*60 + "\n")
#
# 	return {
# 		"validity": validity,
# 		"proximity": avg_proximity,
# 		"fidelity_prob": avg_fidelity_prob,
# 		"sparsity": avg_sparsity,
# 		"successful": successful_cf,
# 		"total": total_graphs,
# 	}

def evaluate_cf_gnnexplainer(pred_model, dataset, device, lr=0.01, epochs=100):
	"""
    评估CF-GNNExplainer
    修改说明：
    1. Validity: 仍然计算翻转成功的比例。
    2. Proximity, Fidelity, Sparsity: 在所有样本上计算平均值（无论是否翻转成功）。
    """
	from torch_geometric.data import Batch, Data
	from torch_geometric.utils import dense_to_sparse
	import numpy as np

	print("\n" + "=" * 60)
	print("Evaluating CF-GNNExplainer (Metrics on ALL samples)")
	print("=" * 60)

	wrapped_model = OracleWrappedModel(pred_model)
	wrapped_model.eval()
	explainer = CFExplainer(wrapped_model, device=device, lr=lr)

	# --- 1. 预计算阶段（不计入 runtime 和 oracle_calls） ---
	y_desired_list = []
	ori_graphs_list = []
	ori_prob_list = []

	print("\n1. Computing original predictions...")
	with torch.no_grad():
		for data in tqdm(dataset, desc="Pre-computing"):
			data = data.to(device)
			ori_pred_logits = pred_model(data.x, data.edge_index, data.batch)
			ori_prob = F.softmax(ori_pred_logits, dim=1)[0]
			ori_pred = ori_pred_logits.argmax(dim=1).item()

			y_desired = 1 - ori_pred

			y_desired_list.append(y_desired)
			ori_graphs_list.append(data)
			ori_prob_list.append(ori_prob)

	# --- 2. 评估循环 ---
	valid_cf = 0
	total_graphs = 0

	proximity_sum = 0.0
	fidelity_prob_sum = 0.0
	sparsity_sum = 0.0

	total_cf_time = 0.0
	total_cf_oracle_calls = 0

	print("\n2. Generating counterfactuals and computing metrics...")

	for idx in tqdm(range(len(dataset)), desc="Evaluating"):
		total_graphs += 1
		ori_data = ori_graphs_list[idx]
		x = ori_data.x
		edge_index = ori_data.edge_index
		batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

		y_desired = y_desired_list[idx]
		ori_prob = ori_prob_list[idx]
		ori_pred = 1 - y_desired

		ori_adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0).to(device)

		# 运行 Explainer 生成 CF —— 仅此阶段计入 runtime 和 oracle_calls
		y_pred_tensor = torch.tensor([ori_pred], device=device)
		calls_before = wrapped_model.oracle_calls
		t0 = time.time()
		cf_adj = explainer.run_one_graph(x, edge_index, batch, ori_adj, y_pred_tensor, epochs)
		total_cf_time += time.time() - t0
		total_cf_oracle_calls += wrapped_model.oracle_calls - calls_before

		cf_edge_index, _ = dense_to_sparse(cf_adj)
		cf_data = Data(
			x=x.cpu(),
			edge_index=cf_edge_index.cpu(),
			num_nodes=x.size(0)
		)

		# --- 检查 Validity（不计入 oracle_calls） ---
		with torch.no_grad():
			cf_data_dev = cf_data.to(device)
			cf_pred_logits = pred_model(
				cf_data_dev.x,
				cf_data_dev.edge_index,
				torch.zeros(cf_data_dev.x.size(0), dtype=torch.long, device=device)
			)
			cf_prob = F.softmax(cf_pred_logits, dim=1)[0]
			cf_pred = cf_pred_logits.argmax(dim=1).item()

		# 只要预测类变成了目标类，就算 Valid
		if cf_pred == y_desired:
			valid_cf += 1

		# ==========================================================
		# 修改：无论是否 Valid，都计算以下指标
		# ==========================================================

		# 1. Proximity（与 MyExplainer 定义保持一致）
		proximity_sum += compute_proximity_from_edge_index(
			ori_edge_index=edge_index,
			cf_edge_index=cf_edge_index,
			num_nodes=x.size(0),
			device=device,
		)

		# 2. Fidelity（概率版）
		fidelity_prob_sum += compute_fidelity_prob_from_probs(
			ori_probs=ori_prob,
			cf_probs=cf_prob,
		)

		# 3. Sparsity
		sparsity_sum += compute_sparsity_from_edge_index(
			ori_edge_index=edge_index,
			cf_edge_index=cf_edge_index,
		)

	avg_runtime_per_graph = total_cf_time / total_graphs if total_graphs > 0 else 0.0
	avg_oracle_calls_per_graph = total_cf_oracle_calls / total_graphs if total_graphs > 0 else 0.0

	# --- 3. 结果汇总 ---
	# Validity: 成功数 / 总数
	validity = valid_cf / total_graphs if total_graphs > 0 else 0.0

	# 其他指标: 总和 / 总数 (不再是 valid_cf)
	if total_graphs > 0:
		avg_proximity = proximity_sum / total_graphs
		avg_fidelity_prob = fidelity_prob_sum / total_graphs
		avg_sparsity = sparsity_sum / total_graphs
	else:
		avg_proximity = 0.0
		avg_fidelity_prob = 0.0
		avg_sparsity = 0.0

	print("\n" + "=" * 60)
	print("Evaluation Results (Calculated on ALL Samples):")
	print("=" * 60)
	print(f"  Validity ↑: {validity:.4f} ({valid_cf}/{total_graphs})")
	print(f"  Proximity (avg all) ↓: {avg_proximity:.4f}")
	print(f"  Fidelity (avg all) ↑: {avg_fidelity_prob:.4f}")
	print(f"  Sparsity (avg all) ↑: {avg_sparsity:.4f}")
	print(f"  Runtime per graph (s) ↓: {avg_runtime_per_graph:.6f}")
	print(f"  Oracle calls per graph ↓: {avg_oracle_calls_per_graph:.4f}")
	print("=" * 60 + "\n")

	return {
		"validity": validity,
		"proximity": avg_proximity,
		"fidelity_prob": avg_fidelity_prob,
		"sparsity": avg_sparsity,
		"successful": valid_cf,
		"total": total_graphs,
		"runtime": avg_runtime_per_graph,
		"oracle_calls": avg_oracle_calls_per_graph,
	}

if __name__ == "__main__":
	dataset_name = os.environ.get("MYEXPLAINER_DATASET", "fluoride_carbonyl")
	device = 'cuda:2'

	print("\n1. Loading datasets...")
	train_dataset, val_dataset, test_dataset = get_datasets(name=dataset_name, root='data/')
	print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

	print("\n2. Loading pre-trained GNN classifier...")
	gnn_path = os.path.join(PROJECT_ROOT, 'param', 'gnns', f'{dataset_name}_gcn.pt')
	gnn = torch.load(gnn_path, map_location=device)
	gnn.eval()
	print("  GNN loaded successfully")

	# 使用新的评估函数
	print("\n3. Evaluating CF-GNNExplainer on validation set...")
	metrics = evaluate_cf_gnnexplainer(
		pred_model=gnn,
		dataset=test_dataset,
		device=device,
		lr=0.01,
		epochs=100  # 每个图优化100步
	)

	print("\nEvaluation completed!")
	print(f"Final Metrics: {metrics}")

