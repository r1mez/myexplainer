import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen')
plt.title("Example Graph")
plt.show()


