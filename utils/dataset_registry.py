"""Central registry for dataset-specific configuration.

Loads dataset entries from configs/dataset_registry.yaml and resolves
class name strings to actual dataset classes from the datasets package.
"""

from pathlib import Path

from datasets import (
    Mutagenicity, MUTAG188, NCI1, AlkaneCarbonyl,
    FluorideCarbonyl, bbbp, BA2Motif, Benzene, PROTEINS,
)
from utils.simple_yaml import load_yaml_file

# Map class name strings (as used in YAML) to actual classes.
_CLASS_MAP = {
    "Mutagenicity": Mutagenicity,
    "MUTAG188": MUTAG188,
    "NCI1": NCI1,
    "AlkaneCarbonyl": AlkaneCarbonyl,
    "FluorideCarbonyl": FluorideCarbonyl,
    "bbbp": bbbp,
    "BA2Motif": BA2Motif,
    "Benzene": Benzene,
    "PROTEINS": PROTEINS,
}

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_registry.yaml"


def _load_registry() -> dict:
    """Load dataset registry from YAML and resolve class references."""
    raw = load_yaml_file(_REGISTRY_PATH)
    datasets = raw.get("datasets", {})
    registry = {}
    for name, entry in datasets.items():
        cls_name = entry["cls"]
        if cls_name not in _CLASS_MAP:
            raise ValueError(
                f"Unknown dataset class '{cls_name}' in registry YAML. "
                f"Available: {', '.join(sorted(_CLASS_MAP))}"
            )
        registry[name] = {
            "cls": _CLASS_MAP[cls_name],
            "folder": entry["folder"],
            "gnn_file": entry["gnn_file"],
            "subgraph": entry["subgraph"],
            "split": entry["split"],
        }
    return registry


DATASET_REGISTRY = _load_registry()


def get_dataset_entry(name: str) -> dict:
    """Get registry entry for a dataset. Raises KeyError if not found."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[key]
