import torch
import torch.nn as nn
import numpy as np
import math
from utils.utils_old import bbox_ious as bbox_iou
from utils.utils_select_device import select_device
from models.bricks.util_loss import IOUloss, IOUWH_loss


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, stride, config_anchor, device_id):#([],80,(w,h))
        super(YOLOLoss, self).__init__()
        #self.depth_maxpool = depth_maxpool
        self.anchors = anchors
        self.total_anchors = self.get_total_anchors(config_anchor)
        self.anchors_mask = self.get_anchors_mask()
        self.num_anchors = len(anchors)#3
        self.free_anchors = 1
        self.num_classes = num_classes#80
        self.bbox_attrs = 5 + num_classes#85
        self.stride = stride
        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1
        self.lambda_cls = 1
        self.mse_loss = nn.MSELoss(reduction='none')#均方误差
        self.bce_loss = nn.BCELoss(reduction='none')#交叉熵
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.iou_whloss = IOUWH_loss()
        self.iou_loss = IOUloss()
        self.device = select_device(device_id)
        in_ch = self.bbox_attrs*self.free_anchors #TODO:这里的in_ch不应该直接生成85类，在yolo_head中不应该直接生成85类！需要更改
        self.guide_wh = nn.Conv2d(in_channels=in_ch,
                                  out_channels=2 * self.free_anchors, kernel_size=1, stride=1, padding=0)
        #这里通过fp直接预测guide_wh注释
        # self.guide_wh = guide_wh_by_depth(self.depth_maxpool, self.free_anchors, in_ch)
        # self.Feature_adaption = nn.Conv2d(in_channels=in_ch,
        #                                   out_channels=in_ch, kernel_size=3, stride=1, padding=1).to(device)
        # self.conv = nn.Conv2d(in_channels=in_ch,
        #                       out_channels=self.free_anchors * (self.num_classes + 5), kernel_size=1, stride=1, padding=0).to(device)

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
        wh_pred = self.guide_wh(input)  # Anchor guiding
        wh_pred = torch.exp(wh_pred)
        # feature_adapt = self.Feature_adaption(input) #TODO:Feature_adaption还没有调好，本来应该加入wh_pred的
        # input = self.conv(feature_adapt)
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.stride
        stride_w = self.stride
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        scaled_total_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.total_anchors]

        prediction = input.view(bs, self.free_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        wh_pred = wh_pred.view(bs, self.free_anchors, 2*self.free_anchors,
                               in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            bs * self.free_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            bs * self.free_anchors, 1, 1).view(y.shape).type(FloatTensor)
        # Calculate anchor w, h
        # anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        # anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        # anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        # anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * wh_pred[..., 0]
        pred_boxes[..., 3] = torch.exp(h.data) * wh_pred[..., 1]

        if targets is not None:
            n_obj, mask, obj_mask, tx, ty, tw, th, tconf, tcls, coord_scale, gussian_weight,t_box = self.get_target(targets, scaled_anchors,
                                                                                                in_w, in_h, pred_boxes
                                                                                                , scaled_total_anchors, wh_pred
                                                                                                )

            mask, obj_mask, coord_scale = mask.to(self.device), obj_mask.to(self.device), coord_scale.to(self.device)
            tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
            tconf, tcls = tconf.to(self.device), tcls.to(self.device)
            gussian_weight = gussian_weight.to(self.device)
            t_box = t_box.to(self.device)

            #  losses #TODO:需要添加IOU_wh_loss,IOU_loss,GIOU_loss,label_smooth,focal_loss
            loss_x = (coord_scale * self.bce_loss(x * mask, tx)).sum()/n_obj
            loss_y = (coord_scale * self.bce_loss(y * mask, ty)).sum()/n_obj
            loss_w = (coord_scale* self.smooth_l1(w * mask , tw )).sum()/n_obj
            loss_h = (coord_scale* self.smooth_l1(h * mask , th )).sum()/n_obj
            #loss_conf = focal_loss(weight_pos=0.3)(obj_mask*conf, tconf).sum()/n_obj
            #loss_conf = focal_loss(gussian_weight)(obj_mask*conf, tconf).sum()/n_obj#这里既加入了yolov3的也加入了gaussian的方法
            loss_conf = self.bce_loss(obj_mask*conf, tconf).sum()/n_obj #这里是原始的conf_loss，这里相当于将0.5的负样本权重改为了1
            default_center = torch.zeros(bs, self.free_anchors, in_h, in_w, 2).to(self.device)
            pred_anchors = torch.cat((default_center, wh_pred),dim=-1)
            loss_wh_iou = self.iou_whloss(pred_anchors[mask==1], t_box[mask==1]).sum()/n_obj#TODO:需要一对一去算loss，不能全部混杂算。
            #loss_iou = self.iou_loss(pred_boxes[mask==1], t_box[mask==1]).sum()/n_obj

            if tcls[mask == 1].shape[0] == 0:
                loss_cls = torch.tensor(0).to(self.device)
            else:
                loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1]).sum()/n_obj

            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + \
                loss_wh_iou

            # print(
            #     'loss:{:.2f},loss_x:{:.5f},loss_y:{:.5f},loss_w:{:.5f},loss_h:{:.5f},loss_conf:{:.5f},loss_cls:{:.2f},loss_wh_iou:{:.2f}'.format(
            #         loss, loss_x * self.lambda_xy, loss_y * self.lambda_xy, loss_w * self.lambda_wh,
            #               loss_h * self.lambda_wh, loss_conf * self.lambda_conf, loss_cls * self.lambda_cls, loss_wh_iou
            #     ))

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_cls.item()#这里返回的只有loss没有item,因为loss还要反向传播
        else:

            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_gaussian_map(self, gi, gj, gw, gh, in_w, in_h, omiga = 2):
        """
        :param gi: the position of the positive sample in x_axis.
        :param gj: the position of the positive sample in y_axis.
        :param gw: the width of the groundtruth box.
        :param gh: the height of the groundtruth box.
        :param in_w: the width of the feature map.
        :param in_h: the height of the feature map.
        :param omiga: the ratio of the width,height to multiple gw, gh.
        for produce the parm of gaussian model.
        :return: the gaussian weight map, shape: [in_w, in_h].
        """
        omiga = omiga
        x_i = torch.arange(in_w).repeat(in_h, 1).float()
        y_j = torch.arange(in_h).repeat(in_w, 1).permute(1, 0).float()
        position_x = torch.ones(x_i.shape)*gi
        position_y = torch.ones(y_j.shape)*gj
        x_direct = torch.pow(x_i - position_x, 2) / (omiga * gw)
        y_direct = torch.pow(y_j - position_y, 2) / (omiga * gh)
        return torch.exp(-0.5*(x_direct + y_direct))

    def get_target(self, target, scaled_anchors, in_w, in_h, pred_boxs, scaled_total_anchors, wh_pred):
        n_obj = 0
        bs = target.shape[0]
        mask = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)  # (bs,3, 13,13)
        scales = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        obj_mask = torch.ones(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, self.free_anchors, in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.free_anchors, in_h, in_w, self.num_classes, requires_grad=False)  # self.num_classes
        gussian_weight = torch.zeros(bs, self.free_anchors, in_h, in_w, requires_grad=False)
        for b in range(bs):
            pred_box = pred_boxs[b].view(-1,4)
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
                # Calculate iou between gt and anchor shapes and choose gt to regress
                anchor_match_ious = bbox_iou(gt_box, anchor_shapes)

                #Calculate iou between  gt_box and predbox
                gt_box[:, :2] = torch.tensor([gx, gy]).to(self.device)
                pred_ious = bbox_iou(gt_box, pred_box).view(self.free_anchors, in_h, in_w)
                obj_mask[b, pred_ious >= self.ignore_threshold] = 1
                obj_mask = 1-obj_mask

                # Find the best matching anchor box
                best_n = torch.argmax(anchor_match_ious)
                # Masks

                if  best_n  in self.anchors_mask:
                # Masks
                    gussian_map = self.get_gaussian_map(gi, gj, gw, gh, in_w, in_h)
                    gussian_weight[b, 0] = torch.max(gussian_map, gussian_weight[b, 0])
                    mask[b, 0, gj, gi] = 1
                    obj_mask[b, 0, gj, gi] = 1
                    scales[b, 0, gj, gi] = 2 - target[b, t, 3] * target[b, t, 4]
                    # Coordinates
                    tx[b, 0, gj, gi] = gx - gi
                    ty[b, 0, gj, gi] = gy - gj
                    # Width and height
                    tw[b, 0, gj, gi] = math.log(
                        gw / wh_pred[b, self.free_anchors-1, gj, gi, 0] + 1e-16)  # 返回去的tw还是要除以选定的anchor的w*in_w
                    # 为什么这里要用math而不是用torch,因为这里只是求真实的值，是用于对比的值，而不是反传回去的值
                    th[b, 0, gj, gi] = math.log(
                        gh / wh_pred[b, self.free_anchors-1, gj, gi, 1] + 1e-16)
                    #t_box
                    t_box[b, 0, gj, gi] = torch.tensor([gx, gy, gw, gh]).float()
                    # object
                    tconf[b, 0, gj, gi] = 1
                    # One-hot encoding of label
                        # one_hot = torch.zeros(self.num_classes, dtype= torch.float32)
                        # one_hot[int(target[b,t,0])] = 1
                        # one_hot_smooth = LabelSmooth()(one_hot, self.num_classes)
                        # tcls[b, anchor_index, gj, gi] = one_hot_smooth
                    tcls[b, 0, gj, gi, int(
                        target[b, t, 0])] = 1  # 这里的target就表明label的类别标签是要从0开始的#int(target([b,t,0])

        return n_obj, mask, obj_mask, tx, ty, tw, th, tconf, tcls, scales, gussian_weight, t_box

if __name__ == '__main__':
    import train.Voc_data_preprocess.params_init_voc as params_init
    config = params_init.TRAINING_PARAMS
    device = select_device(0)
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
