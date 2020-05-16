import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.algorithms.flow import boykov_kolmogorov


def get_layer_data():
    with open('metadata/vgg_layer_info.json') as f:
        data = json.load(f)
    return data


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
        vgg.add_edge("edge", layer_data['layer_name'],
                     capacity=server_time)

        # (3) Add edge processing time to cloud node
        vgg.add_edge(layer_data['layer_name'],
                     "cloud",
                     capacity=layer_data["edge_time"])

    return vgg


def get_partition_node(edge_nodes):
    layer_data = get_layer_data()
    time_data_edge, time_data_server = get_model_layers()
    time_transmission = 0
    time_execution_edge = 0
    time_execution_cloud = 0
    cut_layer = 0

    # calculate the transmission time
    for i in range(1, 24):
        if layer_data[f"{i-1}"] in edge_nodes[0] and layer_data[f"{i}"] in edge_nodes[1]:
            time_transmission = time_data_edge[f'{i-1}']["transmission_time"]
            cut_layer = i

    # Calculate different execution times
    for i in range(1, 24):
        if layer_data[f"{i}"] in edge_nodes[0]:
            time_execution_edge = time_execution_edge + time_data_edge[f"{i}"]["edge_time"]
        elif layer_data[f"{i}"] in edge_nodes[1]:
            time_execution_cloud = time_execution_cloud + time_data_server[f'{i}']["server_time"]

    return edge_nodes[0], edge_nodes[1], time_execution_edge, time_transmission, time_execution_cloud, cut_layer


vgg = build_vgg_graph()

#  Perform a boykov graph-cut between edge and cloud VGG-16 layers
R = boykov_kolmogorov(vgg, 'edge', 'cloud')
flow_value = nx.maximum_flow_value(vgg, 'edge', 'cloud')
print(flow_value)
source_tree, target_tree = R.graph['trees']
execution_nodes = (set(vgg) - set(target_tree), set(target_tree))
print(f'Nodes to be executed on edge : {execution_nodes[0]}')
print(f'Nodes to be executed on cloud : {execution_nodes[1]}')
print(get_partition_node(execution_nodes))

nx.draw(vgg, with_labels=True, font_weight='bold')
plt.draw()
plt.show()
