import torch
import torch.nn as nn
import numpy as np
import math
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_select_device import select_device
from models.bricks.tricks import label_smooth
from models.bricks.tricks import GIOU




class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, stride, config, device_id):#([],80,(w,h))
        super(YOLOLoss, self).__init__()
        config_anchor = config["yolo"]["anchors"]
        self.anchors = anchors
        self.total_anchors = self.get_total_anchors(config_anchor)
        self.anchors_mask = self.get_anchors_mask()
        self.num_anchors = len(anchors)#3
        self.num_classes = num_classes#80
        self.bbox_attrs = 5 + num_classes#85
        self.stride = stride
        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1
        self.lambda_cls = 1
        self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.device = select_device(device_id)
        self.config = config
        self.giou_loss = GIOU()

    def get_anchors_mask(self):
        mask = []
        for anchor in self.anchors:
            mask.append(self.total_anchors.index(anchor))
        return mask
    def get_total_anchors(self,config_anchor):
        total_anchors = []
        for anchor_scale in config_anchor:
            for anchor in anchor_scale:
                total_anchors.append(anchor)
        return total_anchors



    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.stride
        stride_w = self.stride
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        scaled_total_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.total_anchors]

        prediction = input.view(bs, self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pure_cls = prediction[..., 5:]

        #bce多标签多分类
        if self.config['bce']:
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        #ce单标签多分类
        if self.config['ce']:
            pred_cls = torch.softmax(prediction[..., 5:], dim=-1)

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            bs * self.num_anchors, 1, 1).view(x.shape).to(self.device)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            bs * self.num_anchors, 1, 1).view(y.shape).to(self.device)
        # Calculate anchor w, h
        anchor_w = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([0])).to(self.device)
        anchor_h = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([1])).to(self.device)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape).to(self.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        #assign anchor to the feature_map
        # anchor_prior = FloatTensor(prediction[..., :4].shape[1:])
        # anchor_prior[..., 0] = grid_x[0] + 0.5
        # anchor_prior[..., 1] = grid_y[0] + 0.5
        # anchor_prior[..., 2] = anchor_w[0]
        # anchor_prior[..., 3] = anchor_h[0]


        if targets is not None:
            n_obj, mask, noobj_mask, tx, ty, tw, th, tconf, tcls, coord_scale, giou_gt_box= self.get_target(targets, scaled_anchors,
                                                                                                in_w, in_h, pred_boxes
                                                                                                , scaled_total_anchors
                                                                                                )

            mask, noobj_mask, coord_scale = mask.to(self.device), noobj_mask.to(self.device), coord_scale.to(self.device)
            tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
            tconf, tcls = tconf.to(self.device), tcls.to(self.device)
            giou_gt_box = giou_gt_box.to(self.device)

            if self.config["GIOU"]:
                loss_giou = self.giou_loss.cal_giou_loss(giou_gt_box, pred_boxes, mask).sum()/n_obj
            else:
                loss_x = (coord_scale * self.bce_loss(x * mask, tx)).sum()/n_obj
                loss_y = (coord_scale * self.bce_loss(y * mask, ty)).sum()/n_obj
                loss_w = (coord_scale* self.smooth_l1(w * mask , tw )).sum()/n_obj
                loss_h = (coord_scale* self.smooth_l1(h * mask , th )).sum()/n_obj

            loss_conf = (self.bce_loss(conf * mask, mask).sum()/n_obj + \
                         0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0).sum()/n_obj)

            if tcls[mask == 1].shape[0] == 0:
                loss_cls = torch.tensor(0).to(self.device)
            else:
                #加入lable_smooth
                if self.config["label_smooth"]:
                    ls = label_smooth(theta=0.01, classes=self.num_classes)
                    tcls = ls.smooth(tcls, mask)
                if self.config['bce']:
                    loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1]).sum()/n_obj
                #使用多分类交叉熵损失函数
                if self.config['ce']:
                    targets_label = tcls[mask==1]
                    label_rc = torch.where(targets_label==1)
                    #targets_label[label_rc[0], label_rc[1]] = label_rc[1].float()
                    loss_cls = self.ce_loss(pure_cls[mask==1], label_rc[1]).sum()/n_obj

            #  total loss = losses * weight
            if self.config["GIOU"]:
                loss = loss_giou + loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
                # print(
                #     'loss:{:.2f}, loss_giou:{:.5f}, loss_conf:{:.5f},loss_cls:{:.2f}'.format(
                #         loss, loss_giou, loss_conf * self.lambda_conf, loss_cls * self.lambda_cls
                #     ))
                return loss, loss_giou.item(), loss_conf.item(), loss_cls.item()

            else:
                loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                       loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                       loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

                # print(
                #     'loss:{:.2f},loss_x:{:.5f},loss_y:{:.5f},loss_w:{:.5f},loss_h:{:.5f},loss_conf:{:.5f},loss_cls:{:.2f}'.format(
                #         loss, loss_x * self.lambda_xy, loss_y * self.lambda_xy, loss_w * self.lambda_wh,
                #               loss_h * self.lambda_wh, loss_conf * self.lambda_conf, loss_cls * self.lambda_cls
                #     ))
                return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
                       loss_h.item(), loss_conf.item(), loss_cls.item()#这里返回的只有loss没有item,因为loss还要反向传播
        else:

            # Results
            _scale = torch.FloatTensor([stride_w, stride_h] * 2).to(self.device)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, scaled_anchors, in_w, in_h, pred_boxs, scaled_total_anchors):
        n_obj = 0
        bs = target.shape[0]
        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)  # (bs,3, 13,13)
        scales = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)  # self.num_classes
        match_max_IOU = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        giou_gt_box = torch.zeros(bs, self.num_anchors, in_h, in_w, 4, requires_grad=False)

        # anchor_box_prior = torch.zeros()#TODO
        # anchor_box_prior = FloatTensor(prediction[..., :4].shape)
        # anchor_box_prior[..., :2] = torch.floor(pred_box[..., :2])+0.5
        # anchor_box_prior[..., :2] = FloatTensor()


        for b in range(bs):
            pred_box = pred_boxs[b].view(-1, 4)
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                n_obj += 1
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0).to(self.device)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.array([0, 0]*9).reshape(9, 2),
                                                                  np.array(scaled_total_anchors)), 1)).to(self.device)
                # Calculate iou between gt and anchor shapes
                anchor_match_ious = bbox_iou(gt_box, anchor_shapes)

                gt_box[:, :2] = torch.tensor([gx, gy]).to(self.device)
                #gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0).to(device)
                pred_ious = bbox_iou(gt_box, pred_box).view(self.num_anchors, in_h, in_w)
                noobj_mask[b, pred_ious >= self.ignore_threshold] = 0

                # Find the best matching anchor box
                best_n = torch.argmax(anchor_match_ious)

                if best_n in self.anchors_mask:
                    # Masks
                    anchor_index = self.anchors.index(self.total_anchors[best_n])

                    mask[b, anchor_index, gj, gi] = 1
                    noobj_mask[b, anchor_index, gj, gi] = 0

                    # object
                    tconf[b, anchor_index, gj, gi] = 1

                    #这里进行了条件的选择，如果match_iou > 记录的maxiou，则进行更新，但是如果<=且记录的maxiou不为零时，则不进行更新，直接pass,这里的作用是只选取与
                    #anchor交并比匹配最大的真实标注框即gtbox
                    if anchor_match_ious[0][best_n] <= match_max_IOU[b, anchor_index, gj, gi] and match_max_IOU[b, anchor_index, gj, gi] != 0:
                        pass
                    else:
                        scales[b, anchor_index, gj, gi] = 2 - target[b, t, 3] * target[b, t, 4]
                        # Coordinates
                        tx[b, anchor_index, gj, gi] = gx - gi
                        ty[b, anchor_index, gj, gi] = gy - gj

                        # Width and height
                        tw[b, anchor_index, gj, gi] = math.log(
                            gw / scaled_anchors[anchor_index][0] + 1e-16)  # 返回去的tw还是要除以选定的anchor的w*in_w

                        # 为什么这里要用math而不是用torch,因为这里只是求真实的值，是用于对比的值，而不是反传回去的值
                        th[b, anchor_index, gj, gi] = math.log(gh / scaled_anchors[anchor_index][1] + 1e-16)


                        giou_gt_box[b, anchor_index, gj, gi] = gt_box

                        #重置tcls
                        tcls[b, anchor_index, gj, gi] = torch.zeros(self.num_classes, requires_grad=False)
                        tcls[b, anchor_index, gj, gi, int(
                            target[b, t, 0])] = 1  # 这里的target就表明label的类别标签是要从0开始的#int(target([b,t,0])

        return n_obj, mask, noobj_mask, tx, ty, tw, th, tconf, tcls, scales, giou_gt_box

if __name__ == '__main__':
    device = select_device(0)
    import train.params_init_voc as params_init
    config = params_init.TRAINING_PARAMS
    yololoss = YOLOLoss(config["yolo"]["anchors"][0],
                        config["yolo"]["classes"], (416, 416), config_anchor=config["yolo"]["anchors"], device_id=0)
    f = open('../train/output.pkl','rb')
    import pickle
    data = pickle.load(f)
    f.close()
    input = torch.Tensor(data).to(device)
    target_input = [[[0, 0.498, 0.327, 0.997, 0.537]]]
    target_input = np.array(target_input)
    print(yololoss(input, target_input)[0])
    #result: loss_x = 0.0025
    #        loss_y = 0.0036
    #        loss_h = 0.0002
    #        loss_w = 0.0002
    #        loss_conf = 0.3714
    #        loss_cls = 0.6655
