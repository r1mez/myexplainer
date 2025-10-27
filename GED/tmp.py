import torch
from distance_greed import load_neurosed
from datasets import Mutagenicity

dataset = Mutagenicity("../data/mutag", mode="training")
model = load_neurosed(dataset,'best_model.pt', device='cuda:0')
print(model)
print(type(model))