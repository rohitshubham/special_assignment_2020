from torchvision import transforms, models
import torch.nn as nn
import torch
from PIL import Image
import collections


class VGGNet(nn.Module):

    def _conv_block(self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding),
            nn.ReLU(True)
        )

    def _linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.Dropout()
        )

    def __init__(self, num_classes=1000):
        super().__init__()

        self.features_0 = self._conv_block(3,64,3,1,1)
        self.features_2 = self._conv_block(64,64,3,1,1)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_5 = self._conv_block(64,128,3,1,1)
        self.features_7 = self._conv_block(128,128,3,1,1)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_10 = self._conv_block(128,256,3,1,1)
        self.features_12 = self._conv_block(256,256,3,1,1)
        self.features_14 = self._conv_block(256,256,3,1,1)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_17 = self._conv_block(256,512,3,1,1)
        self.features_19 = self._conv_block(512,512,3,1,1)
        self.features_21 = self._conv_block(512,512,3,1,1)
        self.max_pool_2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_24 = self._conv_block(512,512,3,1,1)
        self.features_26 = self._conv_block(512,512,3,1,1)
        self.features_28 = self._conv_block(512,512,3,1,1)
        self.max_pool_2d_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_0 = self._linear_block(512 * 7 * 7, 4096)
        self.classifier_3 = self._linear_block(4096, 4096)
        self.classifier_6 = nn.Linear(4096, num_classes)
        

    def forward(self, x, split=False, split_phase="1"):
        if not split:
            x = self.features_0(x)
            x = self.features_2(x)
            x = self.max_pool_2d_1(x)
            x = self.features_5(x)
            x = self.features_7(x)
            x = self.max_pool_2d_2(x)
            x = self.features_10(x)
            x = self.features_12(x)
            x = self.features_14(x)
            x = self.max_pool_2d_3(x)
            x = self.features_17(x)
            x = self.features_19(x)
            x = self.features_21(x)
            x = self.max_pool_2d_4(x)
    
            x = self.features_24(x)
            x = self.features_26(x)
            x = self.features_28(x)
            x = self.max_pool_2d_5(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            x = self.classifier_0(x)
            x = self.classifier_3(x)
            x = self.classifier_6(x)

        if split:
            if split_phase=="1":
                x = self.features_0(x)
                x = self.features_2(x)
                x = self.max_pool_2d_1(x)
                x = self.features_5(x)
                x = self.features_7(x)
                x = self.max_pool_2d_2(x)
                x = self.features_10(x)
                x = self.features_12(x)
                x = self.features_14(x)
                x = self.max_pool_2d_3(x)
                x = self.features_17(x)
                x = self.features_19(x)
                x = self.features_21(x)
                x = self.max_pool_2d_4(x)
        
                x = self.features_24(x)
                x = self.features_26(x)
                
            elif split_phase=="2":
                x = self.features_28(x)
                x = self.max_pool_2d_5(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)

                x = self.classifier_0(x)
                x = self.classifier_3(x)
                x = self.classifier_6(x)

        return x


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

img = Image.open("/home/rohit/Documents/special_assignment_2020/src/dog.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

test_model.eval()

out = test_model(batch_t,split=True, split_phase="1")
out = test_model(out,split=True, split_phase="2")

with open(r'/home/rohit/Documents/special_assignment_2020/src/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# explain this one
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())
