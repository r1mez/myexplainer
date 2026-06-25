from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from utils.node_labels import (
    NODE_LABEL_CONFIG_PATH,
    _atomic_num_to_symbol_map,
    _dataset_config,
    _load_node_label_config,
    bbbp_atom_symbol_from_feature,
    feature_labels_for_dataset,
    infer_feature_mode,
    infer_node_label,
    infer_node_labels_for_dataset,
    node_label_mode_for_dataset,
    normalize_dataset_name,
)


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


def is_supported_dataset(dataset: str) -> bool:
    return normalize_dataset_name(dataset) in SUPPORTED_DATASETS


def is_supported_split(split: str) -> bool:
    return str(split).strip().lower() in SUPPORTED_SPLITS


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
