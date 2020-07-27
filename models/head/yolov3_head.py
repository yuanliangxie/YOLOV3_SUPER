import torch.nn as nn
from collections import OrderedDict
from .. layer import layer_fundation as layer

class yolov3_head(nn.Module):
    def __init__(self, nAnchors, nClass):
        super().__init__()
        layer_list = [

            OrderedDict([
                ('feature0', layer.Conv2dBatchLeaky(512, 1024, 3, 1)),
                ('reshape', nn.Conv2d(1024, nAnchors*(nClass+5), 1, 1))
            ]),

            OrderedDict([
                ('feature1', layer.Conv2dBatchLeaky(256, 512, 3, 1)),
                ('reshape', nn.Conv2d(512, nAnchors * (nClass + 5), 1, 1))
            ]),

            OrderedDict([
                ('feature2', layer.Conv2dBatchLeaky(128, 256, 3, 1)),
                ('reshape', nn.Conv2d(256, nAnchors * (nClass + 5), 1, 1))
            ])

        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, features):
        output1 = self.layers[0](features[0])
        output2 = self.layers[1](features[1])
        output3 = self.layers[2](features[2])
        output = [output1, output2, output3]
        return output
