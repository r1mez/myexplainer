from .graphxai_mol_dataset import GraphXAIMoleculeDataset


class AlkaneCarbonyl(GraphXAIMoleculeDataset):
    """Balanced GraphXAI alkane-carbonyl dataset with fixed train/val/test splits."""

    dataset_name = "alkane_carbonyl"
    down_sample = True
