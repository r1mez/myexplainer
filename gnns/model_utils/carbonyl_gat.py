import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


def validate_graph_pooling(graph_pooling):
    if graph_pooling not in {"mean", "mean_max"}:
        raise ValueError(f"Unsupported graph_pooling='{graph_pooling}'")


def get_pool_output_dim(hidden_dim, graph_pooling):
    validate_graph_pooling(graph_pooling)
    return hidden_dim if graph_pooling == "mean" else hidden_dim * 2


def pool_graph_representation(x, batch, graph_pooling):
    validate_graph_pooling(graph_pooling)
    mean_pool = global_mean_pool(x, batch)
    if graph_pooling == "mean":
        return mean_pool
    max_pool = global_max_pool(x, batch)
    return torch.cat([mean_pool, max_pool], dim=1)


def get_class_weights(dataset):
    labels = torch.tensor([int(graph.y.item()) for graph in dataset], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2).float()
    class_weights = class_counts.sum() / (class_counts.numel() * class_counts.clamp_min(1.0))
    return class_counts.long(), class_weights


class EdgeWeightedGATConv(GATConv):
    """PyG 2.0.4-compatible GATConv with optional post-attention edge gating."""

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("EdgeWeightedGATConv expects Tensor edge_index when edge_weight is provided.")

        heads = self.heads
        out_channels = self.out_channels

        if isinstance(x, torch.Tensor):
            if x.dim() != 2:
                raise ValueError("Static graphs are expected in 'EdgeWeightedGATConv'.")

            x_src = self.lin_src(x).view(-1, heads, out_channels)
            x_dst = self.lin_dst(x).view(-1, heads, out_channels)
        else:
            x_src, x_dst = x

            if x_src.dim() != 2:
                raise ValueError("Static graphs are expected in 'EdgeWeightedGATConv'.")
            if x_dst is not None and x_dst.dim() != 2:
                raise ValueError("Static graphs are expected in 'EdgeWeightedGATConv'.")

            x_src = self.lin_src(x_src).view(-1, heads, out_channels)
            x_dst = None if x_dst is None else self.lin_dst(x_dst).view(-1, heads, out_channels)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if edge_weight is not None:
            edge_weight = edge_weight.to(device=x_src.device, dtype=x_src.dtype).view(-1)

        if self.add_self_loops:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            if size is not None:
                valid_sizes = [dim for dim in size if dim is not None]
                if valid_sizes:
                    num_nodes = min([num_nodes] + valid_sizes)

            if edge_weight is None:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            else:
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(
                    edge_index,
                    edge_weight,
                    fill_value=1.0,
                    num_nodes=num_nodes,
                )

        out = self.propagate(
            edge_index,
            x=x,
            alpha=alpha,
            edge_attr=None,
            edge_weight=edge_weight,
            size=size,
        )

        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, edge_attr, edge_weight, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = x_j * alpha.unsqueeze(-1)
        if edge_weight is None:
            return out

        # Keep explain-time masks as direct edge gates after attention is computed.
        return out * edge_weight.view(-1, 1, 1)
