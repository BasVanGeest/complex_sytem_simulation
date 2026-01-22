import networkx as nx
import matplotlib.pyplot as plt

SEED = 42
n = 50
m = 2

G = nx.Graph()

BA = nx.barabasi_albert_graph(n, m, SEED)
nx.draw(BA, with_labels=True, node_color="skyblue", node_size=500)
plt.show()