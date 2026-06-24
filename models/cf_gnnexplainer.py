import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import torch.nn as nn
from gnns import *
from torch_geometric.data import InMemoryDataset
from utils import get_datasets
from eval.baseline_eval_metrics import OracleWrappedModel
from models.base import BaseExplainer, CFResult


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

class CFExplainer(BaseExplainer):
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

	def explain_graph(self, data, device="cpu"):
		"""Generate a counterfactual explanation for a single graph.

		Args:
			data: PyG Data object with x, edge_index.
			device: Device string for computation.

		Returns:
			CFResult with cf_edge_index and cf_edge_weight.
		"""
		data = data.to(device)
		x = data.x
		edge_index = data.edge_index
		batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
		adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0).to(device)

		# Get original prediction as the label target
		with torch.no_grad():
			ori_logits = self.pred_model(x, edge_index, batch)
			if ori_logits.dim() > 1:
				ori_pred = ori_logits.argmax(dim=1)[0].item()
			else:
				ori_pred = ori_logits.argmax().item()
		label = torch.tensor([ori_pred], device=device)

		best_cf_adj = self.run_one_graph(x, edge_index, batch, adj, label, epochs=100)
		cf_edge_index, cf_edge_weight = dense_to_sparse(best_cf_adj)

		return CFResult(
			cf_edge_index=cf_edge_index,
			cf_edge_weight=cf_edge_weight,
		)

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

	wrapped_model = OracleWrappedModel(gnn)
	explainer = CFExplainer(wrapped_model, device=device, lr=0.01)

	for idx in range(min(10, len(test_dataset))):
		data = test_dataset[idx]
		result = explainer.explain_graph(data, device=device)
		print(f"Graph {idx}: CF edges={result.cf_edge_index.size(1)}")

