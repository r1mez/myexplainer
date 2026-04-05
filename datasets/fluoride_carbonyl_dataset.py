from .graphxai_mol_dataset import GraphXAIMoleculeDataset


class FluorideCarbonyl(GraphXAIMoleculeDataset):
    """GraphXAI fluoride-carbonyl dataset with fixed train/val/test splits."""

    dataset_name = "fluoride_carbonyl"
    down_sample = False
