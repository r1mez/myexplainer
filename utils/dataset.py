import os

from utils.dataset_registry import get_dataset_entry


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_dataset_root(root):
    if os.path.isabs(root):
        return root
    return os.path.abspath(os.path.join(PROJECT_ROOT, root))


def get_datasets(name, root="data/"):
    """Get preloaded datasets by name."""
    root = _resolve_dataset_root(root)
    entry = get_dataset_entry(name)
    folder = os.path.join(root, entry["folder"])
    cls = entry["cls"]

    print("Loading dataset: ", name)

    if entry["split"] == "slice":
        dataset = cls(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    else:
        train_dataset = cls(folder, mode="training")
        test_dataset = cls(folder, mode="testing")
        val_dataset = cls(folder, mode="evaluation")

    return train_dataset, val_dataset, test_dataset
