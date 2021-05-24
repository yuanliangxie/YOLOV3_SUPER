import torch.nn as nn
import models.head.yolov3_head as yolov3_head
import models.backbone.new_darknet as darknet53
import models.backbone.neck as neck
import models.loss.yolo_loss_baseline_module as loss
from utils.logger import print_logger
from models.layer.layer_fundation import Conv2dBatchLeaky as Convolutional
import numpy as np
import torch
from utils.utils_select_device import select_device


class yolov3(nn.Module):
    def __init__(self, config, logger=None, init_weight=True):
        super().__init__()
        self.backbone = darknet53.darknet53()
        self.neck = neck.neck()
        self.head = yolov3_head.yolov3_head(nAnchors=3, nClass=config["model"]["classes"])
        self.loss = loss.yolo_loss_module(config, strides=[32, 16, 8])
        if logger == None:
            self.logger = print_logger()
        else:
            self.logger = logger
        if init_weight:
            self.__init_weights()

    def forward(self, input, target=None):
        features = self.backbone(input)
        neck_features = self.neck(features)
        yolo_loss_input = self.head(neck_features)
        loss_or_output = self.loss(yolo_loss_input, target)
        return loss_or_output #input=416,[13, 26, 52]

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():#
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                # torch.nn.init.constant_(m.weight.data,0.001)#在测试时为了看模型有没有弄错，进行的改动
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=52):#加载成功
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
            print("weights.shape:{}".format(weights.shape))
        count = 0
        ptr = 0
        for m in self.backbone.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                #conv_layer = m._Convolutional__conv
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d):
                        conv_layer = sub_m
                    elif isinstance(sub_m, nn.BatchNorm2d):
                        bn_layer = sub_m

                # Load BN bias, weights, running mean and running variance
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b

                print("loading weight {}".format(bn_layer))
                # else:
                #     # Load conv. bias
                #     num_b = conv_layer.bias.numel()
                #     conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                #     conv_layer.bias.data.copy_(conv_b)
                #     ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                print("loading weight {}".format(conv_layer))
        print("ptr:{}".format(ptr))
        if ptr == weights.shape[0]:
            print("convert success!")

    # def count_darknet_count(self):
    #     count = 0
    #     for m in self.backbone.modules():
    #         if isinstance(m, Convolutional):
    #             count += 1
    #     print("count:",count)

    def load_darknet_pth_weights(self, pth_file, cutoff=52):
        print("load darknet_coco_pth_weights : ", pth_file)
        count = 0
        pretrain_coco_weight_darknet = torch.load(pth_file)
        list_keys = list(pretrain_coco_weight_darknet.keys())
        keys_count = 0
        for m in self.backbone.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                #conv_layer = m._Convolutional__conv
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d):
                        conv_layer = sub_m
                    elif isinstance(sub_m, nn.BatchNorm2d):
                        bn_layer = sub_m
                if 'conv' in list_keys[keys_count]:
                    weight = pretrain_coco_weight_darknet[list_keys[keys_count]]
                    conv_layer.weight.data.copy_(weight)
                    keys_count +=1

                if 'bn' in list_keys[keys_count]:
                    if "weight" in list_keys[keys_count]:
                        weight = pretrain_coco_weight_darknet[list_keys[keys_count]]
                        bn_layer.weight.data.copy_(weight)
                        keys_count += 1

                    if "bias" in list_keys[keys_count]:
                        bias = pretrain_coco_weight_darknet[list_keys[keys_count]]
                        bn_layer.bias.data.copy_(bias)
                        keys_count += 1

                    if "running_mean" in list_keys[keys_count]:
                        running_mean = pretrain_coco_weight_darknet[list_keys[keys_count]]
                        bn_layer.running_mean.data.copy_(running_mean)
                        keys_count += 1

                    if "running_var" in list_keys[keys_count]:
                        running_var = pretrain_coco_weight_darknet[list_keys[keys_count]]
                        bn_layer.running_var.data.copy_(running_var)
                        keys_count += 1

        print("count:{},keys_count:{}".format(count, keys_count))
        if keys_count == len(list_keys):
            print("convert success!")

if __name__ == '__main__':
    #计算参数量和计算量
    from thop import profile
    from thop import clever_format
    config={"device_id":'cpu', "num_classes":1,
            "model":{"anchors": [[[116, 90], [156, 198], [373, 326]],
                                 [[30, 61], [62, 45], [59, 119]],
                                 [[10, 13], [16, 30], [33, 23]]], "classes":1},
            "ce": False,"bce": True}
    model = yolov3(config)
    input = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")# 增加可读性
    print(flops, params)
    #output = model(input)

