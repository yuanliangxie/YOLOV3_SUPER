import torch
import torch.nn as nn
from .. import layer
import torch.nn.functional as F
import torch.nn.init as init

__all__=['CBL_stack', 'Upsample', 'test_Upsample', 'stack_to_head', 'InvertedResidual', 'make_divisible'
         , 'ressepblock', 'DropBlock', 'pure_upsample', 'ASFF', 'guide_wh_by_depth', 'L2Norm']

class res_Block(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel
        #print(self.input_channel/2)
        self.layer = nn.Sequential(
            layer.Conv2dBatchLeaky(self.input_channel, int(self.input_channel/2), 1, 1),
            layer.Conv2dBatchLeaky(int(self.input_channel/2), self.input_channel, 3, 1)
        )
    def forward(self, x):
        return x + self.layer(x)

class CBL_stack(nn.Module):
    def __init__(self, input_channel, stack_number):
        super(CBL_stack, self).__init__()
        self.input_channel = input_channel
        self.stack_number = stack_number
        block = []
        block.append(layer.Conv2dBatchLeaky(self.input_channel, self.input_channel*2, 3, 2))
        for i in range(self.stack_number):
            block.append(res_Block(input_channel=self.input_channel*2))
        self.layer = nn.Sequential(*block)

    def forward(self, x):
        return self.layer(x)

class Upsample(nn.Module):
    """
    the module is not pure the literal meaning as upsample,but the module is
    used to upsample function in yolov3, it contains two child-modules:
    conv2dBatchLeaky and Upsample

    """
    def __init__(self, input_channel):
        super(Upsample, self).__init__()
        self.input_channel = input_channel
        half_nchannels = int(input_channel / 2)
        layers = [
            layer.Conv2dBatchLeaky(self.input_channel, half_nchannels, 1, 1),
            nn.Upsample(scale_factor=2)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x

def test_Upsample(input_channel):
    half_nchannels = int(input_channel / 2)
    layers = [
        layer.Conv2dBatchLeaky(input_channel, half_nchannels, 1, 1),
        nn.Upsample(scale_factor=2)
    ]
    features = nn.Sequential(*layers)
    return features


class stack_to_head(nn.Module):
    def __init__(self, input_channel, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(input_channel / 2)
        else:
            half_nchannels = int(input_channel / 3)
        in_nchannels = 2 * half_nchannels
        layers = [
            layer.Conv2dBatchLeaky(input_channel, half_nchannels, 1, 1),
            layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
            layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(layer.ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            layer.ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ressepblock(nn.Module):
    def __init__(self, ch, out_ch, in_ch=None, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        in_ch = ch//2 if in_ch==None else in_ch
        resblock_one = nn.ModuleList()
        resblock_one.append(layer.add_conv(ch, in_ch, 1, 1, leaky=False))
        resblock_one.append(layer.add_conv(in_ch, out_ch, 3, 1, leaky=False))
        self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DropBlock(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def reset(self, block_size, keep_prob):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def calculate_gamma(self, x):
        return (1 - self.keep_prob) * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x):
        if (not self.training or self.keep_prob == 1):  # set keep_prob=1 to turn off dropblock
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        if x.type() == 'torch.cuda.HalfTensor':  # TODO: not fully support for FP16 now
            FP16 = True
            x = x.float()
        else:
            FP16 = False
        p = torch.ones_like(x) * (self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        out = mask * x * (mask.numel() / mask.sum())

        if FP16:
            out = out.half()
        return out

class pure_upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(pure_upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

class ASFF(nn.Module):
    #TODO：这里的asff如果加入到模型中则需要self.dim进行更改
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = layer.add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = layer.add_conv(256, self.inter_dim, 3, 2)
            self.expand = layer.add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = layer.add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = layer.add_conv(256, self.inter_dim, 3, 2)
            self.expand = layer.add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = layer.add_conv(512, self.inter_dim, 1, 1)
            self.expand = layer.add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = layer.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = layer.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = layer.add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class guide_wh_by_depth(nn.Module):
    def __int__(self, depth_maxpool, free_anchors, in_ch):
        """
        depth_maxpool: it has been normalized
        """
        #TODO:这里要不要加bn，要不要进行L2_norm
        #TODO:这里的arfa，beta是要单独优化还是要根据网络生成来调?
        super(guide_wh_by_depth,self).__init__()
        self.weight_arfa = nn.Conv2d(in_channels=in_ch,
                              out_channels =2*free_anchors, kernel_size=1, stride=1, padding=0)
        self.bias_beta = nn.Conv2d(in_channels=in_ch,
                              out_channels=2*free_anchors, kernel_size=1, stride=1, padding=0)
        self.depth_maxpool = depth_maxpool

    def forward(self, x):
        arfa = self.weight_arfa(x)
        beta = self.bias_beta(x)
        output = arfa * torch.div(1, self.depth_maxpool) + beta
        return output

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

