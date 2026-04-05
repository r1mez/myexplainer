import os

from datasets import Mutagenicity
from datasets import NCI1
from datasets import AlkaneCarbonyl
from datasets import FluorideCarbonyl
from datasets import bbbp
from datasets import BA2Motif
from datasets import Benzene
from datasets import PROTEINS


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_dataset_root(root):
    if os.path.isabs(root):
        return root
    return os.path.abspath(os.path.join(PROJECT_ROOT, root))


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    root = _resolve_dataset_root(root)
    print("Loading dataset: ", name)
    if name == "mutag":
        folder = os.path.join(root, "mutag")
        train_dataset = Mutagenicity(folder, mode="training")
        test_dataset = Mutagenicity(folder, mode="testing")
        val_dataset = Mutagenicity(folder, mode="evaluation")
    elif name == "nci1":
        folder = os.path.join(root, "NCI1")
        train_dataset = NCI1(folder, mode="training")
        test_dataset = NCI1(folder, mode="testing")
        val_dataset = NCI1(folder, mode="evaluation")
    elif name == "bbbp":
        folder = os.path.join(root, "bbbp")
        dataset = bbbp(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    elif name == "ba2motif":
        folder = os.path.join(root, "ba2motif")
        train_dataset = BA2Motif(folder, mode="training")
        test_dataset = BA2Motif(folder, mode="testing")
        val_dataset = BA2Motif(folder, mode="evaluation")
    elif name == "benzene":
        folder = os.path.join(root, "benzene")
        train_dataset = Benzene(folder, mode="training")
        test_dataset = Benzene(folder, mode="testing")
        val_dataset = Benzene(folder, mode="evaluation")
    elif name == "alkane_carbonyl":
        folder = os.path.join(root, "alkane_carbonyl")
        train_dataset = AlkaneCarbonyl(folder, mode="training")
        test_dataset = AlkaneCarbonyl(folder, mode="testing")
        val_dataset = AlkaneCarbonyl(folder, mode="evaluation")
    elif name == "fluoride_carbonyl":
        folder = os.path.join(root, "fluoride_carbonyl")
        train_dataset = FluorideCarbonyl(folder, mode="training")
        test_dataset = FluorideCarbonyl(folder, mode="testing")
        val_dataset = FluorideCarbonyl(folder, mode="evaluation")
    elif name == "proteins":
        folder = os.path.join(root, "proteins")
        train_dataset = PROTEINS(folder, mode="training")
        test_dataset = PROTEINS(folder, mode="testing")
        val_dataset = PROTEINS(folder, mode="evaluation")
    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
