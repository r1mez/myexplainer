from pathlib import Path

from utils.simple_yaml import load_yaml_file


EXPLAINER_HPARAM_KEYS = (
    "oracle_del_topk",
    "oracle_del_random_negatives",
    "oracle_del_probe_graphs_per_batch",
    "oracle_del_reward_tie_eps",
)

DEFAULT_EXPLAINER_HPARAMS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "explainer_hparams.yaml"
)


def _resolve_config_path(config_path=None):
    if config_path is None or str(config_path).strip() == "":
        return DEFAULT_EXPLAINER_HPARAMS_PATH
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_explainer_hparams(dataset_name, config_path=None):
    dataset_key = str(dataset_name).lower()
    path = _resolve_config_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Explainer hyperparameter config not found: {path}")

    raw_config = load_yaml_file(path)

    datasets = raw_config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Explainer config must contain a top-level 'datasets' mapping: {path}")

    if dataset_key not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(
            f"No explainer hyperparameters configured for dataset '{dataset_key}'. "
            f"Available datasets: {available}"
        )

    params = datasets[dataset_key]
    if not isinstance(params, dict):
        raise ValueError(
            f"Explainer hyperparameters for dataset '{dataset_key}' must be an object."
        )

    missing = [key for key in EXPLAINER_HPARAM_KEYS if key not in params]
    if missing:
        raise ValueError(
            f"Explainer config for dataset '{dataset_key}' is missing keys: {missing}"
        )

    unexpected = sorted(set(params) - set(EXPLAINER_HPARAM_KEYS))
    if unexpected:
        raise ValueError(
            f"Explainer config for dataset '{dataset_key}' has unknown keys: {unexpected}"
        )

    typed_params = {
        "oracle_del_topk": int(params["oracle_del_topk"]),
        "oracle_del_random_negatives": int(params["oracle_del_random_negatives"]),
        "oracle_del_probe_graphs_per_batch": int(params["oracle_del_probe_graphs_per_batch"]),
        "oracle_del_reward_tie_eps": float(params["oracle_del_reward_tie_eps"]),
    }
    return typed_params


def apply_explainer_hparams(args, config_path=None):
    path = config_path
    if path is None:
        path = getattr(args, "explainer_config_path", None)

    params = load_explainer_hparams(getattr(args, "dataset"), path)
    for key, value in params.items():
        setattr(args, key, value)
    return args
