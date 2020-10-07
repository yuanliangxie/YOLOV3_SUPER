from models.loss.yolo_loss import YOLOLoss as YOLOLoss
import torch.nn as nn
class yolo_loss_module(nn.Module):
    def __init__(self, config, strides=[32, 16, 8]):
        super().__init__()
        yolo_losses = []
        for i in range(3):
            yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                        config["yolo"]["classes"], strides[i], config=config, device_id = config["device_id"]))
        # self.yolo_losses_0 = yolo_losses[0]
        # self.yolo_losses_1 = yolo_losses[1]
        # self.yolo_losses_2 = yolo_losses[2]
        self.layers = nn.ModuleList([yolo_losses[i] for i in range(3)])

    def forward(self, input, target=None):
        result = []
        for i in range(3):
            loss_or_output_data = self.layers[i](input[i], target)
            result.append(loss_or_output_data)
        return result
