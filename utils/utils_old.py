from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np



def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions

def bbox_iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def bbox_iou_xywh_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """

    #tranform the xywh to xyxy
    x1 = boxes1[..., 0:1] - boxes1[..., 2:3]/2
    y1 = boxes1[..., 1:2] - boxes1[..., 3:4]/2

    x1_2 = boxes1[..., 0:1] + boxes1[..., 2:3]/2
    y1_2 = boxes1[..., 1:2] + boxes1[..., 3:4]/2

    boxes1 = np.concatenate([x1, y1, x1_2, y1_2], axis=-1)

    x2 = boxes2[..., 0:1] - boxes2[..., 2:3]/2
    y2 = boxes2[..., 1:2] - boxes2[..., 3:4]/2

    x2_2 = boxes2[..., 0:1] + boxes2[..., 2:3]/2
    y2_2 = boxes2[..., 1:2] + boxes2[..., 3:4]/2

    boxes2 = np.concatenate([x2, y2, x2_2, y2_2], axis=-1)




    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., None, :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., None, 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area[:, None] + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


# from utils.utils_select_device import select_device
# device = select_device(0)
# def __convert_pred(pred_bbox, test_input_size, valid_scale, conf_thresh=0.01):
#     """
#     预测框进行过滤，去除尺度不合理的框
#     """
#     pred_bbox = pred_bbox.cpu().numpy()
#     pred_coor = xywh2xyxy(pred_bbox[:, :4])
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]
#
#     # (1)
#     # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
#     # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
#     # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
#     # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
#     # org_h, org_w = org_img_shape
#     # resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
#     # dw = (test_input_size - resize_ratio * org_w) / 2
#     # dh = (test_input_size - resize_ratio * org_h) / 2
#     # pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
#     # pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
#
#     # (2)将预测的bbox中超出原图的部分裁掉
#     pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
#                                 np.minimum(pred_coor[:, 2:], [test_input_size - 1, test_input_size - 1])], axis=-1)
#     # (3)将无效bbox的coor置为0
#     invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
#     pred_coor[invalid_mask] = 0
#
#     # (4)去掉不在有效范围内的bbox
#     bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
#     scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
#
#     # (5)将score低于score_threshold的bbox去掉
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
#     score_mask = scores > conf_thresh
#
#     mask = np.logical_and(scale_mask, score_mask)
#
#     pred_bbox = pred_bbox[mask]
#
#     pred_bbox = torch.Tensor(pred_bbox).to(device)
#
#     return pred_bbox
#
# def non_max_suppression(prediction, num_classes, conf_thres=0.01, nms_thres=0.5):
#     """
#     Removes detections with lower object confidence score than 'conf_thres' and performs
#     Non-Maximum Suppression to further filter detections.
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_score, class_pred)
#     """
#
#     # From (center x, center y, width, height) to (x1, y1, x2, y2)
#     box_corner = prediction.new(prediction.shape)
#     box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
#     box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
#     box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
#     box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
#     prediction[:, :, :4] = box_corner[:, :, :4]
#
#     output = [None for _ in range(len(prediction))]
#     for image_i, image_pred in enumerate(prediction):
#         # Filter out confidence scores below threshold
#         class_max = torch.max(image_pred[:, 5:], dim=-1)[1]
#         score = image_pred[:, 4]*image_pred[:, 5:][torch.Tensor(np.arange(len(image_pred))).long(), class_max]
#         conf_mask = (score >= conf_thres).squeeze()
#         #conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()#TODO:score_thershold
#         image_pred = image_pred[conf_mask]
#         #image_pred = __convert_pred(image_pred, config['img_h'], (0, np.inf))
#
#         # If none are remaining => process next image
#         if not image_pred.size(0):
#             continue
#         # Get score and class with highest confidence
#         max_pred = torch.max(image_pred[:, 5:5+num_classes], 1, keepdim=True)#返回值与索引,这里是取最大的可能的类别，而不会在这个框下一下预测两个框，这个通过用阈值来修改
#         class_conf, class_pred = max_pred
#         # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
#         detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
#         # Iterate through all predicted classes
#         unique_labels = detections[:, -1].unique()
#         if prediction.is_cuda:
#             unique_labels = unique_labels.to(device)
#         temp_detections = []
#         for label in unique_labels:
#             class_detections = detections[detections[:, -1] == label]
#             temp_detections.append(nms_class_detections(class_detections, nms_thres))
#         nms_detections = temp_detections[0]
#         for detection in temp_detections[1:]:
#             nms_detections = torch.cat((nms_detections, detection), 0)
#         output[image_i] = nms_detections.data
#
#     return output
#
# def nms_class_detections(class_detections, nms_thresh):
#     flag = torch.Tensor([True] * class_detections.shape[0]).bool().to(device)
#     _, sort_index = torch.sort(class_detections[:, 4], dim=0, descending=True)
#     class_detections = class_detections[sort_index, :]
#     for i in range(len(flag)):
#         if flag[i] == True:
#             indexs = find_indexTrue_Flag(flag, i)
#             iou = bbox_ious(class_detections[i, :4].unsqueeze(0), class_detections[indexs, :4])
#             mask_iou = iou < nms_thresh
#             flag[indexs] = mask_iou.squeeze()
#         if i == len(flag) - 2:
#             break
#     return class_detections[flag==True, :]
#
# def find_indexTrue_Flag(flag, i):
#     indexs = []
#     for j in range(i+1, len(flag)):
#         index = j
#         if flag[j] == True:
#             indexs.append(index)
#     return indexs


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes

class focal_loss(object):
    """
    loss to balance the positive and negtive
    Args:
        conf: a float tensor of arbitrary shape
              the value of it must be sigmoid or [0,1]
        mask: the label for each element in inputs

    Returns:
        loss tensor with the reduction option applied.
    """
    def __init__(self, gamma=2, weight_pos=-1):
        self.gamma = gamma
        self.weight_pos = weight_pos

    def __call__(self, conf, mask):

        pt = mask * conf + (1-mask) * (1-conf)
        ce_loss = F.binary_cross_entropy(conf, mask, reduction='none')
        if self.weight_pos > 0:
            weight = mask * self.weight_pos + (1-mask) * (1-self.weight_pos)
            #loss = -1 * weight * (1-pt)**self.gamma * torch.log(pt)
            loss = weight * (1-pt)**self.gamma * ce_loss
        else:
            loss = (1-pt)**self.gamma * ce_loss
        return loss

class focal_loss_gaussian_weight(object):
    """
    loss to balance the positive and negtive
    Args:
        initial_args:
            gaussian_weight: calculate for self.weight_neg which is the negtive sample weight
            for the focal loss

        __call__args:
            conf: a float tensor of arbitrary shape
                  the value of it must be sigmoid or [0,1]
            mask: the label for each element in inputs

    Returns:
        loss tensor with the reduction option applied.
    """

    def __init__(self, gaussian_weight, gamma=2, beta=4):
        self.gamma = gamma
        self.beta = beta
        self.weight_pos = 1
        self.weight_neg = torch.pow(1 - gaussian_weight, 4)
        #self.gaussian_weight = gaussian_weight

    def __call__(self, pred_conf_cls, mask):

        pt = mask * pred_conf_cls + (1 - mask) * (1 - pred_conf_cls)
        if self.weight_pos > 0:
            weight = mask * self.weight_pos + (1 - mask) * self.weight_neg
            loss = -1 * weight * (1-pt)**self.gamma * torch.log(pt)
        else:
            loss = (1 - pt) ** self.gamma * torch.log(pt)
        return loss

from torch import nn

class HeatmapLoss(nn.Module):
    def __init__(self,  weight=None, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets):
        #inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0 - inputs) ** self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets) ** self.beta * (inputs) ** self.alpha * torch.log(1.0 - inputs + 1e-14)
        return center_loss + other_loss
if __name__ == '__main__':
    # seed =2
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    #
    # a = torch.sigmoid(torch.randn(2,2))
    # b = torch.ones(2,2)
    # b[1,1] = 0
    # focal = focal_loss(weight_pos=0.2)
    # loss = focal(a, b)
    # print(loss)


    boxes1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    boxes2 = np.array([[0.5, 0.5, 1, 1], [0.5, 0, 1, 1]])
    iou = bbox_iou_xywh_numpy(boxes1, boxes2)
    print(iou)

