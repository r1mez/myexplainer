README for dataset benzene (part of MD-17)

=== Description ===
Dataset MD-17:
Energies and forces from molecular dynamics trajectories of eight organic molecules. Ab initio molecular dynamics trajectories (133k to 993k frames) of benzene, uracil, naphthalene, aspirin, salicylic acid, malonaldehyde, ethanol, toluene at the DFT/PBE+vdW-TS level of theory at 500 K. 

IMPORTANT: The edges present in this dataset were generated using 5A as maximum distance between vertices. Original vertex coordinates were kept, so other cutoffs are still possible, if wanted.

=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Node Label Conversion === 

Node labels were converted to integer values using this map:

Component 0:
	0	C
	1	O
	2	H

=== Node Attributes === 
Node attributes consist of the following values:

<x_coordinate>, <y_coordinate>, <z_coordinate>, <atom_force_x>, <atom_force_y>, <atom_force_z>

=== Graph Attributes === 
Graph attributes correspond to the total energy of the graph.


=== References ===
S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K.Schütt, K.-R. Müller, arXiv:1611.04678 (2017)
https://qmml.org/datasets.html


=== Previous Use of the Dataset ===
 Kristof T. Schütt, Farhad Arbabzadah, Stefan Chmiela, Klaus R. Müller, Alexandre Tkatchenko: Quantum-Chemical Insights from Deep Tensor Neural Networks, Nature Communications 8: 13890, 2017.
 Stefan Chmiela, Alexandre Tkatchenko, Huziel E. Sauceda, Igor Poltavsky, Kristof Schütt, Klaus-Robert Müller: Machine Learning of Accurate Energy-Conserving Molecular Force Fields, Science Advances 3(5): e1603015, 2017.




