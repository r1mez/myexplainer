import os

from utils.loss_hparams import LOSS_HPARAM_KEYS, apply_loss_hparams


BASE_TRAINING_DEFAULTS = {
    "top_k": 1,
    "threshold": 0.0,
    "batch_size": 256,
    "h_dim": 256,
    "z_dim": 32,
    "max_num_nodes": 25,
    "dropout": 0.1,
    "epochs": 1000,
    "lr": 0.01,
    "weight_decay": 1e-5,
    "subgraph_method": "genGraphEx",
    "max_cand_per_graph": 15,
    "candidate_add_threshold": 0.5,
    "hard_keep_threshold": 0.5,
    "hard_add_threshold": 0.5,
    "scheduler_factor": 0.8,
    "scheduler_patience": 15,
    "scheduler_min_lr": 1e-6,
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
    "model_selection": "loss",
    "selection_eval_interval": 25,
    "selection_validity_weight": 1.0,
    "selection_fidelity_weight": 0.5,
    "selection_sparsity_weight": 0.1,
    "selection_proximity_weight": 0.25,
    "train_visualize_batches": 0,
    "eval_visualize_batches": 0,
    "eval_debug_batches": 0,
    "plot_loss_curves": False,
    "print_loss_arrays": False,
}


DATASET_TUNING_PRESETS = {
    "fluoride_carbonyl": {
        # FluorideCarbonyl regressed when we relaxed edit constraints too much.
        # Keep the original operating point and improve checkpoint selection
        # instead of forcing more aggressive edits.
        "batch_size": 256,
        "epochs": 1000,
        "lr": 0.008,
        "weight_decay": 1e-5,
        "max_cand_per_graph": 12,
        "candidate_add_threshold": 0.5,
        "hard_keep_threshold": 0.5,
        "hard_add_threshold": 0.5,
        "scheduler_factor": 0.8,
        "scheduler_patience": 15,
        "scheduler_min_lr": 1e-6,
        "early_stop_patience": 120,
        "early_stop_min_delta": 0.01,
        "model_selection": "composite",
        "selection_eval_interval": 25,
        "selection_validity_weight": 1.0,
        "selection_fidelity_weight": 0.5,
        "selection_sparsity_weight": 0.1,
        "selection_proximity_weight": 0.25,
        "train_visualize_batches": 0,
        "eval_visualize_batches": 0,
        "eval_debug_batches": 0,
        "plot_loss_curves": False,
        "print_loss_arrays": False,
    }
}


def apply_dataset_tuning(args):
    dataset_name = args.dataset.lower()
    defaults = dict(BASE_TRAINING_DEFAULTS)
    if getattr(args, "use_dataset_preset", True):
        defaults.update(DATASET_TUNING_PRESETS.get(dataset_name, {}))

    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    return apply_loss_hparams(args)


def resolve_gnn_checkpoint(dataset_name, gnn_path):
    dataset_name = dataset_name.lower()
    candidates = [
        os.path.join(gnn_path, f"{dataset_name}_gcn.pt"),
        os.path.join(gnn_path, "gnns", f"{dataset_name}_gcn.pt"),
        os.path.join("param", "gnns", f"{dataset_name}_gcn.pt"),
    ]

    checked = []
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in checked:
            continue
        checked.append(abs_candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate

    raise FileNotFoundError(
        "Unable to locate the pre-trained GNN checkpoint for dataset "
        f"'{dataset_name}'. Checked: {checked}. "
        f"If you have not trained it yet, run the corresponding script in "
        f"'gnns/{dataset_name}_gnn.py' first."
    )


def summarize_tuning(args):
    tracked_keys = [
        "dataset",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        *LOSS_HPARAM_KEYS,
        "max_cand_per_graph",
        "candidate_add_threshold",
        "hard_keep_threshold",
        "hard_add_threshold",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_min_lr",
        "early_stop_patience",
        "early_stop_min_delta",
        "model_selection",
        "selection_eval_interval",
        "selection_validity_weight",
        "selection_fidelity_weight",
        "selection_sparsity_weight",
        "selection_proximity_weight",
    ]
    return {key: getattr(args, key, None) for key in tracked_keys}
