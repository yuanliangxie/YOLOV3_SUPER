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
        self.head = yolov3_head.yolov3_head(nAnchors=3, nClass=config["yolo"]["classes"])
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

    # def load_darknet_weights(self, weight_file, cutoff=52):#加载成功
    #     "https://github.com/ultralytics/yolov3/blob/master/models.py"
    #
    #     print("load darknet weights : ", weight_file)
    #
    #     with open(weight_file, 'rb') as f:
    #         _ = np.fromfile(f, dtype=np.int32, count=5)
    #         weights = np.fromfile(f, dtype=np.float32)
    #         print("weights.shape:{}".format(weights.shape))
    #     count = 0
    #     ptr = 0
    #     for m in self.modules():
    #         if isinstance(m, Convolutional):
    #             # only initing backbone conv's weights
    #             if count == cutoff:
    #                 break
    #             count += 1
    #             #conv_layer = m._Convolutional__conv
    #             for sub_m in m.modules():
    #                 if isinstance(sub_m, nn.Conv2d):
    #                     conv_layer = sub_m
    #                 elif isinstance(sub_m, nn.BatchNorm2d):
    #                     bn_layer = sub_m
    #
    #             # Load BN bias, weights, running mean and running variance
    #             num_b = bn_layer.bias.numel()  # Number of biases
    #             # Bias
    #             bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
    #             bn_layer.bias.data.copy_(bn_b)
    #             ptr += num_b
    #             # Weight
    #             bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
    #             bn_layer.weight.data.copy_(bn_w)
    #             ptr += num_b
    #             # Running Mean
    #             bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
    #             bn_layer.running_mean.data.copy_(bn_rm)
    #             ptr += num_b
    #             # Running Var
    #             bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
    #             bn_layer.running_var.data.copy_(bn_rv)
    #             ptr += num_b
    #
    #             print("loading weight {}".format(bn_layer))
    #             # else:
    #             #     # Load conv. bias
    #             #     num_b = conv_layer.bias.numel()
    #             #     conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
    #             #     conv_layer.bias.data.copy_(conv_b)
    #             #     ptr += num_b
    #             # Load conv. weights
    #             num_w = conv_layer.weight.numel()
    #             conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
    #             conv_layer.weight.data.copy_(conv_w)
    #             ptr += num_w
    #             print("loading weight {}".format(conv_layer))
    #     print("ptr:{}".format(ptr))
    #     if ptr == weights.shape[0]:
    #         print("convert success!")
if __name__ == '__main__':
    import train.Voc_data_preprocess.params_init_voc as params_init
    config = params_init.TRAINING_PARAMS
    net = yolov3_mobilev2(config)
    #net.load_darknet_weights('darknet53.conv.74')
    net.eval()
    # dataloader = torch.utils.data.DataLoader(VOCDataset('/home/xyl/Pycharmproject/YOLOv3/YOLOv3_PyTorch/data/voc/trainval.txt',
    #                                                     (config["img_w"], config["img_h"]),
    #                                                     is_training=False, batch_size=1),
    #                                          batch_size=1,
    #                                          shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    #samples = iter(dataloader).__next__()
    #images, labels = samples["image"], samples["label"]
    images = torch.ones((1, 3, 416,416))
    yolo_loss_input = net(images)
    print(yolo_loss_input)

