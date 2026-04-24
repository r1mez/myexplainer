from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils.simple_yaml import load_yaml_file


SUPPORTED_DATASETS = (
    "mutag",
    "mutag188",
    "nci1",
    "bbbp",
    "ba2motif",
    "benzene",
    "alkane_carbonyl",
    "fluoride_carbonyl",
    "proteins",
)

SUPPORTED_SPLITS = ("training", "evaluation", "testing")

NODE_LABEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "node_label_config.yaml"


def normalize_dataset_name(dataset: str) -> str:
    return str(dataset).strip().lower()


def is_supported_dataset(dataset: str) -> bool:
    return normalize_dataset_name(dataset) in SUPPORTED_DATASETS


def is_supported_split(split: str) -> bool:
    return str(split).strip().lower() in SUPPORTED_SPLITS


@lru_cache(maxsize=1)
def _load_node_label_config() -> Dict[str, Any]:
    payload = load_yaml_file(NODE_LABEL_CONFIG_PATH)

    if not isinstance(payload, dict):
        raise ValueError(f"Node label config must be a YAML mapping: {NODE_LABEL_CONFIG_PATH}")
    return payload


def _dataset_config(dataset: str) -> Dict[str, Any]:
    datasets = _load_node_label_config().get("datasets", {})
    if not isinstance(datasets, dict):
        return {}

    dataset_config = datasets.get(normalize_dataset_name(dataset), {})
    return dataset_config if isinstance(dataset_config, dict) else {}


def _config_label_list(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _dataset_feature_labels_from_config(dataset: str, x_dim: int, feature_mode: str) -> Optional[List[str]]:
    config = _dataset_config(dataset)
    if feature_mode == "onehot":
        labels = _config_label_list(config.get("onehot_feature_labels"))
        if labels is not None and len(labels) == x_dim:
            return labels

    labels = _config_label_list(config.get("feature_labels"))
    if labels is not None and len(labels) == x_dim:
        return labels

    return None


def _dataset_node_label_mode_override(dataset: str, x_dim: int) -> Optional[str]:
    config = _dataset_config(dataset)
    mode = config.get("node_label_mode")
    if not isinstance(mode, str) or not mode.strip():
        return None

    min_x_dim = config.get("node_label_min_x_dim", 0)
    try:
        min_x_dim = int(min_x_dim)
    except (TypeError, ValueError):
        min_x_dim = 0

    return mode if x_dim >= min_x_dim else None


def _atomic_num_to_symbol_map(dataset: str) -> Dict[int, str]:
    raw_mapping = _dataset_config(dataset).get("atomic_num_to_symbol")
    if not isinstance(raw_mapping, dict):
        return {}

    mapping = {}
    for key, value in raw_mapping.items():
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _feature_is_onehot(feature: Sequence[float], tol: float = 1e-4) -> bool:
    if len(feature) == 0:
        return False

    total = 0.0
    max_value = -math.inf
    max_count = 0
    for value in feature:
        number = float(value)
        if number < -tol or number > 1.0 + tol:
            return False
        total += number
        if number > max_value + tol:
            max_value = number
            max_count = 1
        elif abs(number - max_value) <= tol:
            max_count += 1

    return abs(total - 1.0) <= tol and max_value >= 1.0 - tol and max_count == 1


def infer_feature_mode(features: Iterable[Sequence[float]]) -> str:
    checked = 0
    for feature in features:
        checked += 1
        if not _feature_is_onehot(feature):
            return "vector"
        if checked >= 64:
            break

    return "onehot" if checked > 0 else "vector"


def feature_labels_for_dataset(dataset: str, x_dim: int, feature_mode: str) -> List[str]:
    dataset = normalize_dataset_name(dataset)
    dataset_labels = _dataset_feature_labels_from_config(dataset, int(x_dim), feature_mode)
    if dataset_labels is not None:
        return dataset_labels

    if feature_mode == "onehot":
        return [str(idx) for idx in range(x_dim)]
    return [f"f{idx}" for idx in range(x_dim)]


def bbbp_atom_symbol_from_feature(feature: Sequence[float]) -> Optional[str]:
    if len(feature) == 0:
        return None

    try:
        atomic_num = int(round(float(feature[0])))
    except (TypeError, ValueError, OverflowError):
        return None

    return _atomic_num_to_symbol_map("bbbp").get(atomic_num)


def node_label_mode_for_dataset(dataset: str, feature_mode: str, x_dim: int) -> str:
    dataset = normalize_dataset_name(dataset)
    override = _dataset_node_label_mode_override(dataset, int(x_dim))
    if override is not None:
        return override
    if feature_mode == "onehot":
        return "onehot"
    return "node_id"


def atom_type_options_for_dataset(dataset: str) -> List[Dict[str, object]]:
    mapping = _atomic_num_to_symbol_map(dataset)
    if not mapping:
        return []

    return [
        {"value": atomic_num, "label": symbol}
        for atomic_num, symbol in sorted(mapping.items())
    ]


def default_atom_type_for_dataset(dataset: str) -> Optional[int]:
    config = _dataset_config(dataset)
    value = config.get("default_atomic_num")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_node_label(
    feature: Sequence[float],
    feature_mode: str,
    feature_labels: Sequence[str],
    node_id: int,
    dataset: Optional[str] = None,
) -> str:
    normalized_dataset = normalize_dataset_name(dataset or "")
    if _dataset_node_label_mode_override(normalized_dataset, len(feature)) == "atomic_num" and len(feature) > 0:
        atom_symbol = bbbp_atom_symbol_from_feature(feature)
        if atom_symbol is not None:
            return atom_symbol
        try:
            return f"Z={int(round(float(feature[0])))}"
        except (TypeError, ValueError, OverflowError):
            pass

    if feature_mode == "onehot" and len(feature) > 0:
        max_idx = max(range(len(feature)), key=lambda idx: float(feature[idx]))
        if 0 <= max_idx < len(feature_labels):
            return str(feature_labels[max_idx])
    return f"n{node_id}"


def infer_node_labels_for_dataset(
    features: Iterable[Sequence[float]],
    dataset: Optional[str] = None,
    feature_mode: Optional[str] = None,
    node_ids: Optional[Iterable[int]] = None,
) -> List[str]:
    feature_rows = [list(feature) for feature in features]
    if not feature_rows:
        return []

    if feature_mode is None:
        feature_mode = infer_feature_mode(feature_rows)

    x_dim = len(feature_rows[0])
    feature_labels = feature_labels_for_dataset(dataset or "", x_dim, feature_mode)
    resolved_node_ids = list(node_ids) if node_ids is not None else list(range(len(feature_rows)))

    return [
        infer_node_label(
            feature,
            feature_mode,
            feature_labels,
            node_id,
            dataset=dataset,
        )
        for node_id, feature in zip(resolved_node_ids, feature_rows)
    ]


def resolve_model_path(
    dataset: str,
    param_root: Path,
    explicit_path: Optional[str] = None,
) -> Optional[Path]:
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.exists():
            return explicit.resolve()
        return None

    dataset = normalize_dataset_name(dataset)
    param_root = Path(param_root).expanduser().resolve()
    search_dirs = []
    if (param_root / "gnns").is_dir():
        search_dirs.append(param_root / "gnns")
    if param_root.is_dir():
        search_dirs.append(param_root)

    expected_lower = f"{dataset}_gcn.pt"

    for directory in search_dirs:
        exact = directory / expected_lower
        if exact.exists():
            return exact.resolve()

        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == expected_lower:
                return entry.resolve()

    return None
