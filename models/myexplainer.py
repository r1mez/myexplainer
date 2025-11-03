import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.inits import glorot, zeros

# from graph_conv import DenseGATConv
from torch_geometric.utils import to_dense_adj

from typing import Optional

class MyExplainer(nn.Module):
    def __init__(self, args,gnn):
        super(MyExplainer, self).__init__()
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.u_dim = args.u_dim
        self.edge_attr_dim = args.edge_attr_dim
        self.max_num_nodes = args.max_subgraph_nodes
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        self.graph_pool_type = 'mean'
        self.device = args.device
        self.model = gnn
        self.device = args.device


        # if self.edge_attr_dim != 0:
        #     self.graph_model = DenseGATConv(self.x_dim, self.edge_attr_dim,self.h_dim)
        # else:
        #     self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)

        # encoder
        # self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + 1 , self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        # self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim , self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

        # decoder
        # self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
        #                                nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
        #                                nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
        # self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
        #                                nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
        #                                nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())
        self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())

        # encoder for target graphs
        # self.encoder_mean_tgt = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        # self.encoder_var_tgt = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(),nn.Sigmoid())

    # def encoder(self, features, adj,y_cf):
    #     graph_rep = self.graph_model(features,adj)  # n x num_node x h_dim
    #     graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
    #
    #     z_mu = self.encoder_mean(torch.cat([graph_rep, y_cf], dim=-1))
    #     z_logvar = self.encoder_var(torch.cat([graph_rep, y_cf], dim=-1))
    #
    #     return z_mu, z_logvar

    def encoder(self, features, adj):
        graph_rep = self.graph_model(features,adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim

        z_mu = self.encoder_mean(torch.cat([graph_rep], dim=-1))
        z_logvar = self.encoder_var(torch.cat([graph_rep], dim=-1))

        return z_mu, z_logvar

    # def encoder_tgt(self, features, adj,y_cf):
    #     graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
    #     graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
    #
    #     z_mu = self.encoder_mean_tgt(torch.cat([graph_rep, y_cf], dim=-1))
    #     z_logvar = self.encoder_var_tgt(torch.cat([graph_rep, y_cf], dim=-1))
    #
    #     return z_mu, z_logvar

    # def decoder(self, z, y_cf):
    #     dec_input = torch.cat([z, y_cf], dim=-1)
    #     x_recon = self.decoder_x(dec_input)
    #     adj_recon = self.decoder_a(dec_input)
    #     return x_recon, adj_recon

    def decoder(self, z, y_cf):
        dec_input = torch.cat([z,y_cf], dim=-1)
        x_recon = self.decoder_x(dec_input)
        adj_recon = self.decoder_a(dec_input)
        return x_recon, adj_recon

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.mean(x, dim=1, keepdim=False)  # 修复：使用mean而不是sum
        return out

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, features, adj, y_cf, features_tgt=None, adj_tgt=None, y=None):
        """
        标准VAE前向传播

        Args:
            features: 输入图的节点特征 (batch, max_num_nodes, x_dim)
            adj: 输入图的邻接矩阵 (batch, max_num_nodes, max_num_nodes)
            y_cf: 目标反事实标签 (batch, 1)
            features_tgt: (可选) 目标图的节点特征，用于配对训练
            adj_tgt: (可选) 目标图的邻接矩阵
            y: (可选) 原始标签

        Returns:
            dict: 包含重构结果和潜在变量参数
        """
        # 1. 编码输入图（使用反事实标签y_cf）
        # z_mu, z_logvar = self.encoder(features, adj, y_cf)
        z_mu, z_logvar = self.encoder(features, adj)

        # 2. 重参数化采样
        z = self.reparameterize(z_mu, z_logvar)

        # 3. 解码生成反事实图
        x_recon, adj_recon = self.decoder(z, y_cf)
        # x_recon, adj_recon = self.decoder(z)

        # 返回基础VAE输出
        output = {
            'x_recon': x_recon,
            'adj_recon': adj_recon,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
        }

        # 4. (可选) 如果提供了目标图，编码用于对比学习
        if features_tgt is not None and adj_tgt is not None and y is not None:
            z_mu_tgt, z_logvar_tgt = self.encoder(features_tgt, adj_tgt, y)
            output['z_mu_tgt'] = z_mu_tgt
            output['z_logvar_tgt'] = z_logvar_tgt

        return output



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h



class DenseGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, aggr='add', bias=True):
        super(DenseGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.aggr = aggr
        # Linear transformations for node features
        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        # Additional transformation for edge attributes
        self.lin_edge = nn.Linear(edge_attr_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, x, adj, edge_attr, mask=None):
        B, N, _ = x.size()
        # Expand edge attributes to match the adjacency matrix
        full_edge_attr = torch.zeros(B, N, N, self.edge_attr_dim, device=edge_attr.device)
        tril_indices = torch.tril_indices(N, N, offset=-1)
        full_edge_attr[:, tril_indices[0], tril_indices[1]] = edge_attr
        full_edge_attr[:, tril_indices[1], tril_indices[0]] = edge_attr
        # Transform edge attributes
        edge_attr_transformed = self.lin_edge(full_edge_attr).squeeze(-1)  # Shape: [B, N, N]

        # Modify adjacency matrix with edge attributes
        edge_adj = adj * edge_attr_transformed  # Shape: [B, N, N]

        # Perform graph convolution
        out = torch.matmul(edge_adj, x)  # Shape: [B, N, out_channels]
        if self.aggr == 'mean':
            out = out / edge_adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        out = self.lin_rel(out)
        out += self.lin_root(x)
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, edge_attr_dim={self.edge_attr_dim})')
