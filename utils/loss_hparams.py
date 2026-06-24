from pathlib import Path

from utils.simple_yaml import load_yaml_file


LOSS_HPARAM_KEYS = (
    "w_cf",
    "w_l1_add",
    "w_l1_del",
    "w_oracle_del_rank",
    "w_vgae_recon",
    "w_vgae_kl",
    "enable_fs_feature_recon",
    "w_vgae_feat_recon",
    "w_proto",
    "cf_margin",
    "lambda_cf_margin",
)

DEFAULT_LOSS_HPARAMS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "loss_hparams.yaml"
)


def _resolve_config_path(config_path=None):
    if config_path is None or str(config_path).strip() == "":
        return DEFAULT_LOSS_HPARAMS_PATH
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_loss_hparams(dataset_name, config_path=None):
    dataset_key = str(dataset_name).lower()
    path = _resolve_config_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Loss hyperparameter config not found: {path}")

    raw_config = load_yaml_file(path)

    datasets = raw_config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Loss config must contain a top-level 'datasets' mapping: {path}")

    if dataset_key not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(
            f"No loss hyperparameters configured for dataset '{dataset_key}'. "
            f"Available datasets: {available}"
        )

    params = datasets[dataset_key]
    if not isinstance(params, dict):
        raise ValueError(
            f"Loss hyperparameters for dataset '{dataset_key}' must be an object."
        )

    missing = [key for key in LOSS_HPARAM_KEYS if key not in params]
    if missing:
        raise ValueError(
            f"Loss config for dataset '{dataset_key}' is missing keys: {missing}"
        )

    unexpected = sorted(set(params) - set(LOSS_HPARAM_KEYS))
    if unexpected:
        raise ValueError(
            f"Loss config for dataset '{dataset_key}' has unknown keys: {unexpected}"
        )

    typed_params = {}
    for key in LOSS_HPARAM_KEYS:
        value = params[key]
        if key == "enable_fs_feature_recon":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Loss config value '{key}' for dataset '{dataset_key}' must be true/false."
                )
            typed_params[key] = value
        else:
            typed_params[key] = float(value)
    return typed_params
