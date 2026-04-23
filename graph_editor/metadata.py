from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


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

MUTAG_NODE_LABELS = [
    "C",
    "O",
    "Cl",
    "H",
    "N",
    "F",
    "Br",
    "S",
    "P",
    "I",
    "Na",
    "K",
    "Li",
    "Ca",
]

MUTAG188_NODE_LABELS = [
    "C",
    "N",
    "O",
    "F",
    "I",
    "Cl",
    "Br",
]

GRAPHXAI_MOL_ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "Na",
    "Ca",
    "I",
    "B",
    "H",
    "*",
]

BBBP_ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    15: "P",
    16: "S",
    17: "Cl",
    20: "Ca",
    35: "Br",
    53: "I",
}

BBBP_DEFAULT_ATOMIC_NUM = 6

BBBP_FEATURE_LABELS = [
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_hs",
    "num_radical_electrons",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
]


def normalize_dataset_name(dataset: str) -> str:
    return str(dataset).strip().lower()


def is_supported_dataset(dataset: str) -> bool:
    return normalize_dataset_name(dataset) in SUPPORTED_DATASETS


def is_supported_split(split: str) -> bool:
    return str(split).strip().lower() in SUPPORTED_SPLITS


def _feature_is_onehot(feature: Sequence[float], tol: float = 1e-4) -> bool:
    if not feature:
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
    if dataset == "bbbp" and x_dim == len(BBBP_FEATURE_LABELS):
        return list(BBBP_FEATURE_LABELS)
    if feature_mode == "onehot":
        if dataset == "mutag" and x_dim == len(MUTAG_NODE_LABELS):
            return list(MUTAG_NODE_LABELS)
        if dataset == "mutag188" and x_dim == len(MUTAG188_NODE_LABELS):
            return list(MUTAG188_NODE_LABELS)
        if dataset in {"benzene", "alkane_carbonyl", "fluoride_carbonyl"} and x_dim == len(GRAPHXAI_MOL_ATOM_TYPES):
            return list(GRAPHXAI_MOL_ATOM_TYPES)
        return [str(idx) for idx in range(x_dim)]
    return [f"f{idx}" for idx in range(x_dim)]


def bbbp_atom_symbol_from_feature(feature: Sequence[float]) -> Optional[str]:
    if not feature:
        return None

    try:
        atomic_num = int(round(float(feature[0])))
    except (TypeError, ValueError, OverflowError):
        return None

    return BBBP_ATOMIC_NUM_TO_SYMBOL.get(atomic_num)


def node_label_mode_for_dataset(dataset: str, feature_mode: str, x_dim: int) -> str:
    dataset = normalize_dataset_name(dataset)
    if dataset == "bbbp" and x_dim >= 1:
        return "atomic_num"
    if feature_mode == "onehot":
        return "onehot"
    return "node_id"


def atom_type_options_for_dataset(dataset: str) -> List[Dict[str, object]]:
    dataset = normalize_dataset_name(dataset)
    if dataset != "bbbp":
        return []

    return [
        {"value": atomic_num, "label": symbol}
        for atomic_num, symbol in sorted(BBBP_ATOMIC_NUM_TO_SYMBOL.items())
    ]


def default_atom_type_for_dataset(dataset: str) -> Optional[int]:
    dataset = normalize_dataset_name(dataset)
    if dataset == "bbbp":
        return BBBP_DEFAULT_ATOMIC_NUM
    return None


def infer_node_label(
    feature: Sequence[float],
    feature_mode: str,
    feature_labels: Sequence[str],
    node_id: int,
    dataset: Optional[str] = None,
) -> str:
    if normalize_dataset_name(dataset or "") == "bbbp" and feature:
        atom_symbol = bbbp_atom_symbol_from_feature(feature)
        if atom_symbol is not None:
            return atom_symbol
        try:
            return f"Z={int(round(float(feature[0])))}"
        except (TypeError, ValueError, OverflowError):
            pass

    if feature_mode == "onehot" and feature:
        max_idx = max(range(len(feature)), key=lambda idx: float(feature[idx]))
        if 0 <= max_idx < len(feature_labels):
            return str(feature_labels[max_idx])
    return f"n{node_id}"


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
