import torch.nn as nn
import torch


class VGGNet(nn.Module):
    """
    A custom implementation of VGG-16 originally published at:
    Simonyan, Karen & Zisserman, Andrew. (2014). Very Deep Convolutional
    Networks for Large-Scale Image Recognition. arXiv 1409.1556.

    This implementation uses Pytorch framework and has total of
    23 layers. It allows partial inference and returns the intermediate tensor
    value.
    ...

    Attributes
    ----------
    num_classes : int
        The integer value of number of labels/classes.
        This defines the output linear tensor size.

    Methods
    -------
    forward(tensor, start_layer = 0, stop_layer = 22)
        Performs the inference. If the start and stop parameters are not
        provided, it performs complete inference.
    """

    def _conv_block(self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding),
            nn.ReLU(True)
        )

    def _get_dictionary_layers(self):
        _layers_dict = {0: self.features_0,
                        1: self.features_2,
                        2: self.max_pool_2d_1,
                        3: self.features_5,
                        4: self.features_7,
                        5: self.max_pool_2d_2,
                        6: self.features_10,
                        7: self.features_12,
                        8: self.features_14,
                        9: self.max_pool_2d_3,
                        10: self.features_17,
                        11: self.features_19,
                        12: self.features_21,
                        13: self.max_pool_2d_4,
                        14: self.features_24,
                        15: self.features_26,
                        16: self.features_28,
                        17: self.max_pool_2d_5,
                        18: self.avgpool,
                        19: torch.flatten,
                        20: self.classifier_0,
                        21: self.classifier_3,
                        22: self.classifier_6
                        }

        return _layers_dict

    def _linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.Dropout()
        )

    def __init__(self, num_classes=1000):
        super().__init__()

        self.features_0 = self._conv_block(3, 64, 3, 1, 1)
        self.features_2 = self._conv_block(64, 64, 3, 1, 1)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_5 = self._conv_block(64, 128, 3, 1, 1)
        self.features_7 = self._conv_block(128, 128, 3, 1, 1)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_10 = self._conv_block(128, 256, 3, 1, 1)
        self.features_12 = self._conv_block(256, 256, 3, 1, 1)
        self.features_14 = self._conv_block(256, 256, 3, 1, 1)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_17 = self._conv_block(256, 512, 3, 1, 1)
        self.features_19 = self._conv_block(512, 512, 3, 1, 1)
        self.features_21 = self._conv_block(512, 512, 3, 1, 1)
        self.max_pool_2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_24 = self._conv_block(512, 512, 3, 1, 1)
        self.features_26 = self._conv_block(512, 512, 3, 1, 1)
        self.features_28 = self._conv_block(512, 512, 3, 1, 1)
        self.max_pool_2d_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_0 = self._linear_block(512 * 7 * 7, 4096)
        self.classifier_3 = self._linear_block(4096, 4096)
        self.classifier_6 = nn.Linear(4096, num_classes)

    def forward(self, input_tensor, start_layer=0, stop_layer=22):
        """Performs the inference.

        If the start and stop parameters are not provided,
        it performs complete inference.

        Parameters
        ----------
        input_tensor : pytorch tensor object,
            The Partial/Initial tensor object

        start_layer : int, optional
            The start layer of execution

        stop_layer : int, optional
            The end layer of partial execution

        Raises
        ------
        """

        layers = self._get_dictionary_layers()
        for i in range(start_layer, stop_layer + 1):
            if i != 19:
                input_tensor = layers[i](input_tensor)
            else:
                input_tensor = layers[i](input_tensor, 1)

        return input_tensor
