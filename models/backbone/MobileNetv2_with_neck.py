import torch
import torch.nn as nn
from models import bricks as bricks
from models import layer as layer
class MobileNetv2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """

        super(MobileNetv2, self).__init__()
        block = bricks.InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            # only check the first element, assuming user knows t,c,n,s are required
            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
                raise ValueError("inverted_residual_setting should be non-empty "
                                 "or a 4-element list, got {}".format(inverted_residual_setting))

            # building first layer
            input_channel = bricks.make_divisible(input_channel * width_mult, round_nearest)
            last_channel = bricks.make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            mlist = nn.ModuleList()
            mlist.append(layer.ConvBNReLU(3, input_channel, stride=2))
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                output_channel = bricks.make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    mlist.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            # building last several layers
            mlist.append(layer.ConvBNReLU(input_channel, last_channel, kernel_size=1))  # 18

            # YOLOv3
            mlist.append(bricks.ressepblock(last_channel, 1024, in_ch=512, shortcut=False))  # 19
            mlist.append(layer.add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1, leaky=False))  # 20
            # SPP Layer
            mlist.append(layer.SPPLayer())  # 21

            mlist.append(layer.add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1, leaky=False))  # 22
            mlist.append(layer.add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1, leaky=False))  # 23
            mlist.append(bricks.DropBlock(block_size=1, keep_prob=1))  # 24
            mlist.append(layer.add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1, leaky=False))  # 25 (17)

            # 1st yolo branch
            mlist.append(layer.add_conv(in_ch=512, out_ch=256, ksize=1, stride=1, leaky=False))  # 26
            mlist.append(bricks.pure_upsample(scale_factor=2, mode='nearest'))  # 27
            mlist.append(layer.add_conv(in_ch=352, out_ch=256, ksize=1, stride=1, leaky=False))  # 28
            mlist.append(layer.add_conv(in_ch=256, out_ch=512, ksize=3, stride=1, leaky=False))  # 29
            mlist.append(bricks.DropBlock(block_size=1, keep_prob=1))  # 30
            mlist.append(bricks.ressepblock(512, 512, in_ch=256, shortcut=False))  # 31
            mlist.append(layer.add_conv(in_ch=512, out_ch=256, ksize=1, stride=1, leaky=False))  # 32
            # 2nd yolo branch

            mlist.append(layer.add_conv(in_ch=256, out_ch=128, ksize=1, stride=1, leaky=False))  # 33
            mlist.append(bricks.pure_upsample(scale_factor=2, mode='nearest'))  # 34
            mlist.append(layer.add_conv(in_ch=160, out_ch=128, ksize=1, stride=1, leaky=False))  # 35
            mlist.append(layer.add_conv(in_ch=128, out_ch=256, ksize=3, stride=1, leaky=False))  # 36
            mlist.append(bricks.DropBlock(block_size=1, keep_prob=1))  # 37
            mlist.append(bricks.ressepblock(256, 256, in_ch=128, shortcut=False))  # 38
            mlist.append(layer.add_conv(in_ch=256, out_ch=128, ksize=1, stride=1, leaky=False))  # 39

            self.module_list = mlist
            self.level_0_conv = layer.add_conv(in_ch=512, out_ch=512, ksize=3, stride=1, leaky=False)
            self.level_1_conv = layer.add_conv(in_ch=256, out_ch=256, ksize=3, stride=1, leaky=False)
            self.level_2_conv = layer.add_conv(in_ch=128, out_ch=128, ksize=3, stride=1, leaky=False)

    def forward(self, x):
        route_layers = []
        for i, module in enumerate(self.module_list):

            # yolo layers
            x = module(x)
            #print(x.shape)
            # route layers
            if i in [6, 13, 25, 32, 39]:
                route_layers.append(x)
            if i == 27:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 34:
                x = torch.cat((x, route_layers[0]), 1)

        # first output the largest rception
        stack_to_head1 = self.level_0_conv(route_layers[2])

        # first output the middle rception
        stack_to_head2 = self.level_1_conv(route_layers[3])

        # first output the smallest rception
        stack_to_head3 = self.level_2_conv(route_layers[4])

        features = [stack_to_head1, stack_to_head2, stack_to_head3]

        return features

if __name__ == '__main__':
    input = torch.randn(1, 3 ,416,416)
    mobilenet2 = MobileNetv2()
    features = mobilenet2(input)
    for i in features:
        print(i.shape)