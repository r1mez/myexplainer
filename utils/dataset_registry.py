"""Central registry for dataset-specific configuration.

Each dataset has a single entry that captures:
- Dataset class and folder name
- GNN checkpoint path
- Subgraph mining parameters
- Split strategy
"""

from datasets import (
    Mutagenicity, MUTAG188, NCI1, AlkaneCarbonyl,
    FluorideCarbonyl, bbbp, BA2Motif, Benzene, PROTEINS,
)


DATASET_REGISTRY = {
    "mutag": {
        "cls": Mutagenicity,
        "folder": "mutag",
        "gnn_file": "mutag_gcn.pt",
        "subgraph": {"method": "discrete", "N": 417, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "mutag188": {
        "cls": MUTAG188,
        "folder": "mutag188",
        "gnn_file": "mutag188_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "nci1": {
        "cls": NCI1,
        "folder": "NCI1",
        "gnn_file": "nci1_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "bbbp": {
        "cls": bbbp,
        "folder": "bbbp",
        "gnn_file": "bbbp_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "slice",
    },
    "ba2motif": {
        "cls": BA2Motif,
        "folder": "ba2motif",
        "gnn_file": "ba2motif_gcn.pt",
        "subgraph": {"method": "continuous", "N": 25, "num_samples": 50, "threshold": 0.97},
        "split": "standard",
    },
    "benzene": {
        "cls": Benzene,
        "folder": "benzene",
        "gnn_file": "benzene_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "alkane_carbonyl": {
        "cls": AlkaneCarbonyl,
        "folder": "alkane_carbonyl",
        "gnn_file": "alkane_carbonyl_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "fluoride_carbonyl": {
        "cls": FluorideCarbonyl,
        "folder": "fluoride_carbonyl",
        "gnn_file": "fluoride_carbonyl_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
    "proteins": {
        "cls": PROTEINS,
        "folder": "proteins",
        "gnn_file": "proteins_gcn.pt",
        "subgraph": {"method": "discrete", "N": 111, "num_samples": 100, "threshold": 0.1},
        "split": "standard",
    },
}


def get_dataset_entry(name: str) -> dict:
    """Get registry entry for a dataset. Raises KeyError if not found."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[key]
