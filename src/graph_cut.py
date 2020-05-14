import networkx as nx
import matplotlib.pyplot as plt
import json


def get_model_layers():
    with open('metadata/layer_metadata.json') as f:
        data = json.load(f)
    return data


def build_vgg_graph():
    data = get_model_layers()
    vgg = nx.DiGraph()
    for i in range(1, 23):
        layer_data = data[f'{i}']
        layer_data_next = data[f'{i + 1}']
        vgg.add_edge(layer_data['layer_name'],
                     layer_data_next['layer_name'],
                     capacity=layer_data['transmission_time'])
    return vgg

# Do the boykov edge split
# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.boykov_kolmogorov.html#networkx.algorithms.flow.boykov_kolmogorov


vgg = build_vgg_graph()
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

nx.draw(vgg, with_labels=True, font_weight='bold')
plt.draw()
plt.show()
