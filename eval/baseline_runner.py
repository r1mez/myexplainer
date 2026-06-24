"""Shared evaluation harness for baseline counterfactual explainer models."""
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from eval.metrics import proximity, fidelity, sparsity
from models.base import BaseExplainer


class BaselineRunner:
    """Run evaluation for any baseline counterfactual explainer.

    Supports two usage patterns:

    1. BaseExplainer interface (recommended):
        runner = BaselineRunner(explainer, gnn, config)
        results = runner.run(data_loader)

    2. Custom forward function:
        runner = BaselineRunner(model, gnn, config)
        results = runner.run(data_loader, model_forward_fn=my_forward)
    """

    def __init__(self, model, gnn, config):
        self.model = model
        self.gnn = gnn
        self.config = config

    @staticmethod
    def _explainer_forward_fn(explainer: BaseExplainer, device: str):
        """Create a model_forward_fn from a BaseExplainer instance.

        The returned callable takes (model, batch) and returns a list of
        Data objects representing counterfactual graphs.
        """
        def forward_fn(model, batch):
            if isinstance(batch, Batch):
                data_list = batch.to_data_list()
            elif isinstance(batch, Data):
                data_list = [batch]
            else:
                data_list = batch

            cf_list = []
            for data in data_list:
                result = explainer.explain_graph(data, device=device)
                cf_data = Data(
                    x=data.x.cpu(),
                    edge_index=result.cf_edge_index.cpu(),
                    edge_weight=result.cf_edge_weight.cpu(),
                    num_nodes=data.x.size(0),
                )
                cf_list.append(cf_data)
            return cf_list

        return forward_fn

    def run(self, data_loader, model_forward_fn=None):
        """Run evaluation and return metrics dict.

        Args:
            data_loader: DataLoader for evaluation data
            model_forward_fn: callable(model, batch) -> cf_graphs
                             If None and self.model is a BaseExplainer,
                             uses explainer.explain_graph() automatically.
                             Otherwise uses model(batch).

        Returns:
            dict with validity, proximity, fidelity, sparsity
        """
        if model_forward_fn is None:
            if isinstance(self.model, BaseExplainer):
                device = str(self.config.device) if hasattr(self.config, 'device') else "cpu"
                model_forward_fn = self._explainer_forward_fn(self.model, device)
            else:
                model_forward_fn = lambda m, b: m(b)

        self.model.eval()
        self.gnn.eval()

        total = 0
        valid = 0
        prox_sum = 0.0
        fid_sum = 0.0
        spars_sum = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating baseline"):
                # Get original graphs
                if isinstance(batch, dict):
                    ori_graphs = batch['graphs'].to(self.config.device)
                else:
                    ori_graphs = batch.to(self.config.device)

                # Get original predictions
                ori_pred_logits, _ = self.gnn.get_pred(
                    ori_graphs.x, ori_graphs.edge_index, ori_graphs.batch
                )
                ori_prob = torch.softmax(ori_pred_logits, dim=1)
                ori_pred = ori_pred_logits.argmax(dim=1)
                y_desired = (1 - ori_pred).long().unsqueeze(1)

                # Generate CF graphs
                cf_graphs = model_forward_fn(self.model, ori_graphs)

                # Ensure cf_graphs is a Batch object
                if isinstance(cf_graphs, list):
                    cf_graphs = Batch.from_data_list(cf_graphs)
                elif not isinstance(cf_graphs, Batch):
                    # Single Data object, wrap in list and batch
                    cf_graphs = Batch.from_data_list([cf_graphs])

                # Compute metrics
                batch_size = ori_graphs.num_graphs if hasattr(ori_graphs, 'num_graphs') else 1
                total += batch_size

                # Validity
                cf_pred_logits, _ = self.gnn.get_pred(
                    cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch
                )
                cf_pred = cf_pred_logits.argmax(dim=1).view(-1, 1)
                valid += (cf_pred == y_desired).sum().item()

                # Other metrics
                prox_sum += proximity(self.config, cf_graphs, ori_graphs)
                fid_sum += fidelity(self.config, ori_graphs, cf_graphs, ori_prob, self.gnn)
                spars_sum += sparsity(self.config, ori_graphs, cf_graphs)

        return {
            "validity": valid / total if total > 0 else 0.0,
            "proximity": prox_sum / total if total > 0 else 0.0,
            "fidelity": fid_sum / total if total > 0 else 0.0,
            "sparsity": spars_sum / total if total > 0 else 0.0,
            "valid": valid,
            "total": total,
        }
