import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer
from models import bricks as bricks


class neck(nn.Module):
    def __init__(self):
        super().__init__()
        layer_list = [
            # layer3 - > 0
            # output the layer recep_largest
            OrderedDict([
                ('stack_to_head1', bricks.stack_to_head(1024, first_head=True))
            ]),

            # layer4 -> 1
            # output the layer recep middle

            OrderedDict([
                ('stack_to_head2', bricks.stack_to_head(768, first_head=False))
            ]),

            # layer5 -> 2
            # output the layer recep small

            OrderedDict([
                ('stack_to_head3', bricks.stack_to_head(384, first_head=False))
            ]),

            # layer6 -> 3
            # UpSample, connect the reception_largest  to reception_second

            OrderedDict([
                ('upsample1', bricks.Upsample(512))
            ]),

            # layer7 -> 4
            # UpSample, connect the reception_middle to reception_small
            OrderedDict([
                ('upsample2', bricks.Upsample(256))
            ])

        ]
        self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, input):
        # stage4 = self.layer[0](input)
        # stage5 = self.layer[1](stage4)
        # stage6 = self.layer[2](stage5)
        # # first output the largest rception
        stage4, stage5, stage6 = input
        stack_to_head1 = self.layer[0](stage6)

        upsample1 = self.layer[3](stack_to_head1)
        concat1 = torch.cat([upsample1, stage5], 1)

        # second output the middle reception
        stack_to_head2 = self.layer[1](concat1)

        upsample2 = self.layer[4](stack_to_head2)
        concat2 = torch.cat([upsample2, stage4], 1)
        # third output the smallest rception
        stack_to_head3 = self.layer[2](concat2)
        features = [stack_to_head1, stack_to_head2, stack_to_head3]#input=416,[13, 26, 52]

        return features
