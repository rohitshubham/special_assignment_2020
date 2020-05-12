import networkx as nx
import matplotlib.pyplot as plt
import json


def get_model_layers():
    with open('metadata/vgg_layer_info.json') as f:
        data = json.load(f)
    return data


def build_vgg_graph(data):
    data = get_model_layers()
    vgg = nx.DiGraph()
    for i in range(23):

G = nx.Graph()
G.add_node(1)
G.add_edge(3, 2)
G.add_edge(1, 2)
print(G.number_of_nodes())
print(G.edges([2]))

FG = nx.DiGraph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n, nbrs in FG.adj.items():
    for nbr, eattr in nbrs.items():
        wt = eattr['weight']
        print('(%d, %d, %.3f)' % (n, nbr, wt))

plt.subplot(121)
nx.draw(FG, with_labels=True, font_weight='bold')
plt.show()
