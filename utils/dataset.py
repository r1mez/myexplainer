import os

from datasets import Mutagenicity
from datasets import NCI1
from datasets import bbbp
from datasets import BA2Motif
from datasets import Benzene

def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
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
    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
