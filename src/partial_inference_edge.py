import torch
import collections
from vgg import VGGNet
import json
from torchvision import transforms
from grpc_client import send_grpc_msg
from PIL import Image


device = torch.device('cpu')

model_path = r"/home/rohit/vgg16-397923af.pth"

test_model = VGGNet(1000)

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


def partial_inference(img):
    transform = transforms.Compose([transforms.Resize(256),	
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        out = test_model(batch_t, start_layer=0, stop_layer=13)
        est = json.dumps(out.tolist())
        out = send_grpc_msg(est, 14)
    return out


out = partial_inference(Image.open("images/dog.jpg"))
print(out)
