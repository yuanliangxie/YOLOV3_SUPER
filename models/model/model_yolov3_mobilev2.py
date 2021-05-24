import torch.nn as nn
import models.head.yolov3_head as yolov3_head
import models.backbone.MobileNetv2_with_neck as MobileNetv2
import models.loss.yolo_loss_baseline_module as loss
from utils.logger import print_logger
import torch

class yolov3_mobilev2(nn.Module):
    def __init__(self, config, logger=None, init_weight=True):
        super().__init__()
        self.backbone = MobileNetv2.MobileNetv2()
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
        yolo_loss_input = self.head(features)
        loss_or_output = self.loss(yolo_loss_input, target)
        return loss_or_output #input=416,[13, 26, 52]

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
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

if __name__ == '__main__':
    #计算参数量和计算量
    from thop import profile
    from thop import clever_format
    config={"device_id":'cpu', "num_classes":1,
            "model":{"anchors": [[[116, 90], [156, 198], [373, 326]],
                                 [[30, 61], [62, 45], [59, 119]],
                                 [[10, 13], [16, 30], [33, 23]]], "classes":1},
            "ce": False, "bce": True}
    model = yolov3_mobilev2(config)
    input = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")# 增加可读性
    print(flops, params)

