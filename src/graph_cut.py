import networkx as nx
import matplotlib.pyplot as plt
import json


def get_model_layers():
    with open('metadata/layer_metadata.json') as f:
        data = json.load(f)
    with open('metadata/server_time.json') as f:
        server_time_data = json.load(f)
    return data, server_time_data


def build_vgg_graph():
    data, server_time_data = get_model_layers()
    vgg = nx.DiGraph()

    # Add input layer
    vgg.add_edge(data["0"]['layer_name'],
                 data["1"]["layer_name"],
                 capacity=data["0"]['transmission_time'])

    for i in range(1, 24):
        layer_data = data[f'{i}']
        server_time = server_time_data[f'{i}']["server_time"]
        # As per equation 4 of the original paper
        # we construct the graph
        if i != 23:
            # (1) Add transmission time as weight
            layer_data_next = data[f'{i + 1}']
            vgg.add_edge(layer_data['layer_name'],
                         layer_data_next['layer_name'],
                         capacity=layer_data['transmission_time'])
       
        # (2) Add cloud processing time to edge node
        vgg.add_edge("edge",
                     layer_data['layer_name'],
                     capacity=server_time)

        # (3) Add edge processing time to cloud node
        vgg.add_edge("cloud",
                     layer_data['layer_name'],
                     capacity=layer_data["edge_time"])

    return vgg

# Do the boykov edge split
# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.boykov_kolmogorov.html#networkx.algorithms.flow.boykov_kolmogorov


vgg = build_vgg_graph()

nx.draw(vgg, with_labels=True, font_weight='bold')
plt.draw()
plt.show()
