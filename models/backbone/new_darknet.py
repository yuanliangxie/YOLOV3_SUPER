import torch
import torch.nn as nn
from collections import OrderedDict
from models import layer as layer
from models import bricks as bricks


class darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_Channel = 32
        stage_cfg = {'stage_2': 1, 'stage_3': 2, 'stage_4': 8, 'stage_5': 8, 'stage_6': 4}
        layer_list = [
            # layer 0
            # first layer, smallest_reception
            OrderedDict([
                ('stage1', layer.Conv2dBatchLeaky(3, 32, 3, 1)),
                ('stage2', bricks.CBL_stack(32, stage_cfg['stage_2'])),
                ('stage3', bricks.CBL_stack(64, stage_cfg['stage_3'])),
                ('stage4', bricks.CBL_stack(128, stage_cfg['stage_4'])),
            ]),
            # layer 1
            # second layer, larger_reception
            OrderedDict([
                ('stage5', bricks.CBL_stack(256, stage_cfg['stage_5']))
            ]),
            # layer 2
            # third layer, largest_reception
            OrderedDict([
                ('stage6', bricks.CBL_stack(512, stage_cfg['stage_6']))
            ]),

            # # layer3
            # # output the layer recep_largest
            # OrderedDict([
            #     ('stack_to_head1', bricks.stack_to_head(1024, first_head=True))
            # ]),
            #
            # # layer4
            # # output the layer recep middle
            #
            # OrderedDict([
            #     ('stack_to_head2', bricks.stack_to_head(768, first_head=False))
            # ]),
            #
            # # layer5
            # # output the layer recep small
            #
            # OrderedDict([
            #     ('stack_to_head3', bricks.stack_to_head(384, first_head=False))
            # ]),
            #
            # # layer6
            # # UpSample, connect the reception_largest  to reception_second
            #
            # OrderedDict([
            #     ('upsample1', bricks.Upsample(512))
            # ]),
            #
            # # layer7
            # # UpSample, connect the reception_middle to reception_small
            # OrderedDict([
            #     ('upsample2', bricks.Upsample(256))
            # ])

        ]
        self.layer = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, input):
        stage4 = self.layer[0](input)
        stage5 = self.layer[1](stage4)
        stage6 = self.layer[2](stage5)
        # first output the largest rception
        # stack_to_head1 = self.layer[3](stage6)
        #
        # upsample1 = self.layer[6](stack_to_head1)
        # concat1 = torch.cat([upsample1, stage5], 1)
        #
        # # second output the middle reception
        # stack_to_head2 = self.layer[4](concat1)
        #
        # upsample2 = self.layer[7](stack_to_head2)
        # concat2 = torch.cat([upsample2, stage4], 1)
        # # third output the smallest rception
        # stack_to_head3 = self.layer[5](concat2)
        # features = [stack_to_head1, stack_to_head2, stack_to_head3]#input=416,[13, 26, 52]
        features = [stage4, stage5, stage6]
        return features

if __name__ == '__main__':
    input = torch.randn(1, 3, 416, 416)
    darknet = darknet53()
    features = darknet(input)
    for i in features:
        print(i.shape)
