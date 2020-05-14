from torchvision import transforms
import torch
from PIL import Image
import collections
from vgg import VGGNet
import json
import time

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(dev)

model_path = r"/home/rohit/vgg16-397923af.pth"
test_model = VGGNet(1000)
test_model.to(device)

loaded_model_weights = torch.load(model_path)

bandwidth = 6.71  # in Mbits per/second

modified_weights = collections.OrderedDict()
for layer_name, weights in loaded_model_weights.items():
    new_layer_name = layer_name.replace(".", "_", 1)
    if "classifier_6" not in new_layer_name:
        new_layer_name = new_layer_name.replace(".", ".0.", 1)
    modified_weights[new_layer_name] = weights

test_model.load_state_dict(modified_weights)

test_model.eval()

skip_server_time_generation = False


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_model_layers():
    with open('metadata/vgg_layer_info.json') as f:
        data = json.load(f)
    return data


def set_layer_metadata(layer_metadata, file_name):
    with open(f'metadata/{file_name}.json', 'w') as json_file:
        json.dump(layer_metadata, json_file)


def detect_images(img):
    transform = transforms.Compose([transforms.Resize(256),	
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    data = get_model_layers()
    layer_metadata = {}

    def generate_edge_parameters(out):
        # input image size
        print(f'{data[str(0)]} : {(out.element_size() * out.nelement())/1024/1024} MB')
        layer_metadata[0] = {"layer_name": data[str(0)], "size": (out.element_size() * out.nelement())/1024/1024}

        for i in range(0, 23):
            start_time_edge = time.time()
            out = test_model(out, start_layer=i, stop_layer=i)
            elapsed_time = time.time() - start_time_edge
            size = (out.element_size() * out.nelement())/1024/1024  #in MB
            print(f'{data[str(i+1)]} : {size} MB, time = {elapsed_time} ')
            layer_metadata[i+1] = {"layer_name": data[str(i+1)],
                                   "size": size,
                                   "edge_time": elapsed_time,
                                   "transmission_time": size/bandwidth}

        set_layer_metadata(layer_metadata, "layer_metadata")
        return out

    def generate_server_time(out):
        layer_metadata[0] = {"layer_name": data[str(0)], "server_time": 0}
        print(f'{data[str(0)]} : time = {0}')

        for i in range(0, 23):
            start_time_edge = time.time()
            out = test_model(out, start_layer=i, stop_layer=i)
            elapsed_time = time.time() - start_time_edge
            print(f'{data[str(i+1)]} : time = {elapsed_time}')
            layer_metadata[i+1] = {"layer_name": data[str(i+1)],
                                   "server_time": elapsed_time}

        set_layer_metadata(layer_metadata, "server_time")
        return out

    with torch.no_grad():
        if not skip_server_time_generation:
            out = generate_edge_parameters(batch_t)            
        else:
            out = generate_server_time(batch_t)
    return out


out = detect_images(Image.open("images/dog.jpg"))

labels = load_classes("classes/imagenet_classes.txt")


_, index = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

for idx in range(1):
    print(f"Image: {labels[index[0][idx]]}, Confidence: {percentage[index[0][idx]].item()}")
