import torch
import json
import collections
import yaml
from dnn_architectures.vgg import VGGNet
from torchvision import transforms
from grpc_client import send_grpc_msg
from PIL import Image
from graph_cut.graph_cut import partition_light


def get_layer_data():
    """Returns the layer information for VGG-16"""
    with open('metadata/vgg_layer_info.json') as f:
        data = json.load(f)
    return data


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


print(f"Reading configuration to load model")
model_configuration = load_configuration()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')


model_path = model_configuration['path']
print(f"{model_configuration['architecture']} pre-trained model loaded")

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

    Parameters:
    path (string): String to the model path for VGG-16
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


test_model.load_state_dict(modified_weights)

test_model.eval()


def get_partition_layer():
    """
    Returns the partition layer information after 
    performing the Dynamic Surgery Light (DSL) algorithm 
    """
    layer_info = get_layer_data()
    print("Attempting to perform the graph cut")
    result = partition_light()
    print(f"The partition layer is {layer_info[f'{result[6]}']}({result[6]})")
    return result[6]


def partial_inference(img):
    """
    Perform the partial inference at edge and calls server using gRPC to complete the inference.
    Prints the inference output.
    """
    partition_layer = get_partition_layer()
    transform = transforms.Compose([transforms.Resize(256),	
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        print(f"Starting VGG-16 inference")
        out = test_model(batch_t, start_layer=0, stop_layer=partition_layer-1)
        print(f"Executed the code till layer {partition_layer-1}. Sending to cloud now.")
        intermediate_tensor_list = out.tolist()
        out = send_grpc_msg(intermediate_tensor_list, partition_layer)
    return out


out = partial_inference(Image.open("images/dog.jpg"))
print(out)
