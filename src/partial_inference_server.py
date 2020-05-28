import torch
import collections
import yaml
from dnn_architectures.vgg import VGGNet
import time


def load_configuration():
    """
    Reads the configuration.yaml file for model configuration
    """
    configuration = {}
    with open("configuration.yaml", 'r') as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return configuration['model']


model_configuration = load_configuration()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)


model_path = model_configuration['path']

test_model = VGGNet(model_configuration['classes'])
test_model.to(device)

loaded_model_weights = torch.load(model_path)

modified_weights = collections.OrderedDict()
for layer_name, weights in loaded_model_weights.items():
    new_layer_name = layer_name.replace(".", "_", 1)
    if "classifier_6" not in new_layer_name:
        new_layer_name = new_layer_name.replace(".", ".0.", 1)
    modified_weights[new_layer_name] = weights


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


test_model.load_state_dict(modified_weights)

test_model.eval()


def partial_inference(intermediate_data, start_layer):
    """
    Perform the partial inference at edge and calls server using gRPC to complete the inference.
        
    Paramters:
    intermediate_data (list): the intermediate inference data recieved from edge device
    start_layer (int) : the layer form which to begin inference. 
    """
    out = torch.Tensor(intermediate_data).to(device)
    with torch.no_grad():
        start_time = time.time()
        out = test_model(out, start_layer=start_layer, stop_layer=22)
        print(f"Finished processing request from layer {start_layer} in {time.time() - start_time}")
    labels = load_classes("classes/imagenet_classes.txt")

    _, index = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    inference = f"Image: {labels[index[0][0]]}, Confidence: {percentage[index[0][0]].item()}"
    return inference
