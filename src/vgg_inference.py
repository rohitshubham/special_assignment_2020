from torchvision import transforms
import torch
from PIL import Image
import collections
from vgg import VGGNet

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

test_model.load_state_dict(modified_weights)

transform = transforms.Compose([
    transforms.Resize(256),	transforms.CenterCrop(224),	transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

img = Image.open("/home/rohit/Documents/Special Assignment/src/dog.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

test_model.eval()

out = test_model(batch_t, start_layer=0, stop_layer=15)

out = test_model(out, start_layer=16, stop_layer=22)

with open(r'/home/rohit/Documents/Special Assignment/src/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# explain this one
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(f"Image: {labels[index[0]]}, Confidence: {percentage[index[0]].item()}")
