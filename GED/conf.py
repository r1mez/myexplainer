from omegaconf import OmegaConf, DictConfig


def get_conf() -> DictConfig:
    # 定义配置字典
    config = {
        'task': {
            'name': 'GRAPHEDX_xor_on_edge_mutagenicity__2025-07-08||15:54:25',
            'wandb_project': None,
            'wandb_group': None
        },
        'log': {
            'dir': 'no_attr_logs'
        },
        'base_dir': '.',
        'training': {
            'batch_size': 256,
            'device': 'cuda:1',
            'dropout': 0,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'num_epochs': 2000,
            'seed': 0,
            'run_till_early_stopping': True,
            'wandb_watch': False,
            'overwrite': False,
            'patience': 100,
            'weights_dir': 'no_attr_weights',
            'sinkhorn_temp': 0.01,
            'sinkhorn_noise': 0
        },
        'dataset': {
            'name': 'mutagenicity',
            'path': 'no_attr_data',
            'data_type': 'gmn',
            'return_adj': True,
            'node_ins_cost': 1,
            'node_del_cost': 1,
            'node_rel_cost': 0,
            'edge_ins_cost': 1,
            'edge_del_cost': 1,
            'edge_rel_cost': 0,
            'one_hot_dim': 1,
            'max_set_size': 20,
            'max_node_set_size': 20,
            'max_edge_set_size': 23,
            'num_classes': 35
        },
        'model': {
            'name': 'GRAPHEDX_xor_on_edge',
            'classPath': 'models.graphedx',
            'LAMBDA': 0.1,
            'output_mode': 'L1',
            'edge_scale': 1,
            'use_max': False,
            'use_second_sinkhorn': False,
            'use_second_sinkhorn_log': False,
            'use_h_hp_node': False,
            'use_m_ms_edge': True
        },
        'gmn': {
            'filters_3': 10,
            'GMN_NPROPLAYERS': 5,
            'variant': 'deep'
        },
        'mode': 'no_attr',
        'src': {
            'train_graphedx': None
        },
        'method_name': 'XOR_AD',
        '\\': None,
        'data_mode': 'equal'
    }

    # 使用 OmegaConf 创建 DictConfig 对象
    return OmegaConf.create(config)

def get_gmn_conf() -> dict:
    # 定义配置字典
    config = {
        'encoder': {
            'node_hidden_sizes': [10, 10],
            'node_feature_dim': 1,
            'edge_hidden_sizes': None,
            'edge_feature_dim': 1
        },
        'aggregator': {
            'node_hidden_sizes': [10, 10],
            'graph_transform_sizes': [10],
            'input_size': [10],
            'gated': True,
            'aggregation_type': 'sum'
        },
        'graph_embedding_net': {
            'node_state_dim': 10,
            'edge_hidden_sizes': [20],
            'node_hidden_sizes': [10, 10],
            'n_prop_layers': 5,
            'share_prop_params': True,
            'edge_net_init_scale': 0.1,
            'node_update_type': 'gru',
            'use_reverse_direction': True,
            'reverse_dir_param_different': False,
            'layer_norm': False,
            'prop_type': 'embedding'
        },
        'graph_matching_net': {
            'node_state_dim': 10,
            'edge_hidden_sizes': [20],
            'node_hidden_sizes': [10, 10],
            'n_prop_layers': 5,
            'share_prop_params': True,
            'edge_net_init_scale': 0.1,
            'node_update_type': 'gru',
            'use_reverse_direction': True,
            'reverse_dir_param_different': False,
            'layer_norm': False,
            'prop_type': 'matching',
            'similarity': 'dotproduct'
        },
        'model_type': 'embedding',
        'data': {
            'problem': 'graph_edit_distance',
            'dataset_params': {
                'n_nodes_range': [20, 20],
                'p_edge_range': [0.2, 0.2],
                'n_changes_positive': 1,
                'n_changes_negative': 2,
                'validation_dataset_size': 1000
            }
        },
        'training': {
            'batch_size': 256,
            'learning_rate': 0.0001,
            'mode': 'pair',
            'loss': 'margin',
            'margin': 1.0,
            'graph_vec_regularizer_weight': 1e-06,
            'clip_value': 10.0,
            'n_training_steps': 500000,
            'print_after': 100,
            'eval_after': 10
        },
        'evaluation': {
            'batch_size': 256
        },
        'seed': 0,
        'graphsim': {
            'conv_kernel_size': [10, 6, 4, 2],
            'linear_size': [40, 10],
            'gcn_size': [10, 10, 10, 10, 10],
            'conv_pool_size': [3, 3, 2, 2],
            'conv_out_channels': [2, 4, 6, 8],
            'dropout': 0
        }
    }

    return config



# 示例用法
if __name__ == "__main__":
    conf = get_conf()
    print(conf)
    print(type(conf))  # 验证类型为 DictConfig

    test_gmn_config = get_gmn_conf()
    print(test_gmn_config)
    print(type(test_gmn_config))  # 验证类型为 dict