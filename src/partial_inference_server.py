import torch
import collections
from vgg import VGGNet
import json

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device('cpu')

model_path = r"/home/rohit/vgg16-397923af.pth"

test_model = VGGNet(1000)
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
    out = torch.Tensor(json.loads(intermediate_data)).to(device)
    with torch.no_grad():
        out = test_model(out, start_layer=start_layer, stop_layer=22)
    labels = load_classes("classes/imagenet_classes.txt")

    _, index = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    inference = f"Image: {labels[index[0][0]]}, Confidence: {percentage[index[0][0]].item()}"
    return inference
