import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from dataset import ssd_augmentations

# mean = np.array([0.40789654, 0.44719302, 0.47026115],
#                     dtype=np.float32).reshape(1, 1, 3)
# std = np.array([0.28863828, 0.27408164, 0.27809835],
#                    dtype=np.float32).reshape(1, 1, 3)
mean = (103, 113, 119)


class DetracDataset(Dataset):
    def __init__(self, list_path, ignore_region_path, labels_path, img_size, is_training, is_debug=False, batch_size=16):
        self.img_files = []
        self.label_files = []
        self.batch_size = batch_size
        self.mean = [103, 113, 119]
        for path in open(list_path, 'r'): #'vehecal/labels'
            path_split = path.split('/')
            index_jpg = path_split[-1][3:]
            video_name = path_split[-2]
            label_path = os.path.join(labels_path, video_name+index_jpg.replace('.jpg', '.txt')).strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                print("no label found. skip it: {}".format(path))
        # 提取视频目录下的掩模
        self.ignore_region = dict()
        for ignore_region_line in open(ignore_region_path):
            line_split = ignore_region_line.strip().split(" ")
            key = line_split[0]
            len_box = (len(line_split) - 1) / 4
            remainder = len_box - int(len_box)
            assert remainder == 0, "ignore_region标签有误！"
            values = []
            for i in range(int(len_box)):
                ignore_box = line_split[1+4*i:5+4*i]
                x1, y1, w, h = [float(coor) for coor in ignore_box]
                x2 = x1 + w
                y2 = y1 + h
                values.append([int(x1), int(y1), int(x2), int(y2)])
            self.ignore_region[key] = values

        self.is_debug = is_debug

        #  transforms and augmentation
        self.total_transforms = ssd_augmentations.Compose(transforms=[])

        # self.transforms.add(data_transforms.KeepAspect())

        if is_training:
            self.total_transforms.add(ssd_augmentations.SSDAugmentation(mean=mean))
            if img_size is None:
                self.total_transforms.add(ssd_augmentations.ResizeImage_multi_scale(batch_size=self.batch_size, mean=mean))
            else:
                self.total_transforms.add(ssd_augmentations.ResizeImage_single_scale(new_size=img_size, mean=mean))
        else:
            self.total_transforms.add(ssd_augmentations.ToAbsoluteCoords())
            self.total_transforms.add(ssd_augmentations.xywh_to_xyxy())
            self.total_transforms.add(ssd_augmentations.ResizeImage_single_scale(new_size=img_size, mean=mean))
        self.total_transforms.add(ssd_augmentations.xyxy_to_xywh())
        self.total_transforms.add(ssd_augmentations.ToPercentCoords())
        self.total_transforms.add(ssd_augmentations.ToTensor(self.is_debug))

    def index_2_classes(self, classes):
        index = [i for i in range(len(classes))]
        index_2_class = dict(zip(index, classes))
        return index_2_class
    def classes_2_index(self, classes):
        index = [i for i in range(len(classes))]
        class_2_index = dict(zip(classes, index))
        return class_2_index

    def __getitem__(self, index):
        #print('index='+str(index))
        img_path = self.img_files[index].rstrip()
        img_path_split = img_path.split('/')
        video_name = img_path_split[-2]
        ignore_boxes = self.ignore_region[video_name]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#加入训练的数据是RGB格式的

        ori_h, ori_w = img.shape[:2]
        ###############加入掩模##########################################
        for boxes in ignore_boxes:
            x1, y1, x2, y2 = boxes
            image_mask = np.ones((y2 - y1, x2 - x1, 3)).astype(np.uint8)
            for i in range(3):
                image_mask[:, :, i] = image_mask[:, :, i] * mean[i]
                img[y1:y2, x1:x2, :] = image_mask


        label_path = self.label_files[index].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 8)#np.loadtxt可以直接加载文本中具有固定格式的文字数据
            assert labels.shape[0] != 0
        else:
            print("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 8), np.float32)
        image = img
        box = labels[:, 1:5]
        label = labels[:, 0].reshape(-1, 1)
        # truncation_ratio = labels[:, 5].reshape(-1, 1)
        # overlap_ratio = labels[:, 6].reshape(-1, 1)
        # iou_ratio = labels[:, 7].reshape(-1, 1)

        if self.total_transforms is not None:
            image, box, label = self.total_transforms(image, box, label)

        #######################################下面可以添加label中除了bbox，label的其他属性用于可视化###############3
        # truncation_ratio = torch.from_numpy(truncation_ratio).type(torch.FloatTensor)
        # overlap_ratio = torch.from_numpy(overlap_ratio).type(torch.FloatTensor)
        # iou_ratio = torch.from_numpy(iou_ratio).type(torch.FloatTensor)
        #label = torch.cat((label, box, truncation_ratio, overlap_ratio, iou_ratio), 1)
        label = torch.cat((label, box), 1)
        img_path_split = img_path.split('/')
        img_ind = img_path_split[-2]+img_path[-9:-4]
        origin_size = [ori_h, ori_w]
        return image, label, img_ind, origin_size

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        '''Pad images and encode targets.

            As for images are of different sizes, we need to pad them to the same size.

            Args:
              batch: (list) of images, cls_targets, loc_targets.

            Returns:
              padded images, stacked cls_targets, stacked loc_targets.

            Reference:
              https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/blob.py
            '''
        batchsize = len(batch)
        image = [x[0] for x in batch]
        label = [x[1] for x in batch]
        img_ind = [x[2] for x in batch]
        origin_size = [x[3] for x in batch]
        max_objects = max([x.shape[0] for x in label])
        label_new = []
        for i in range(batchsize):
            label_len = label[i].shape[0]
            #填满label标签
            filled_labels = torch.zeros(max_objects, label[0].shape[1]).type(torch.FloatTensor)
            filled_labels[range(max_objects)[:label_len]] = label[i][:]
            label_new.append(filled_labels)
        sample = {'image': torch.stack(image), 'label': torch.stack(label_new), 'img_ind': img_ind, "origin_size": origin_size}
        return sample


#  use for test dataloader
if __name__ == "__main__":
    import sys
    MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
    import train.Detrac_data_preprocess.params_init_Detrac as params_init
    classes_category = params_init.TRAINING_PARAMS["yolo"]["classes_category"]
    vocdataset = DetracDataset(list_path="../data/detrac/test.txt", ignore_region_path="../data/detrac/test_ignore_region.txt",
                            labels_path='../data/detrac/labels_test',
                               img_size=(960, 960), is_training=False, is_debug=True, batch_size=8)
    index_2_classes = vocdataset.index_2_classes(classes_category)
    dataloader = torch.utils.data.DataLoader(vocdataset,
                                             batch_size=8,
                                             shuffle=False, num_workers=0, pin_memory=False, collate_fn=vocdataset.collate_fn)
    print(len(vocdataset))
    for step, sample in enumerate(dataloader):
        #print(step)
        for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
            image = image.numpy()
            label = label.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                # truncation_ratio = l[5]
                # overlap_ratio = l[6]
                # iou_ratio = l[7]
                # if truncation_ratio > np.array([0.95], dtype=np.float32)[0]:
                #     print("{:.3f}".format(truncation_ratio))
                #     print("truncation_ratio没有过滤好")
                # if overlap_ratio > np.array([0.95], dtype=np.float32)[0]:
                #     print("{:.3f}".format(overlap_ratio))
                #     print("overlap_ratio没有过滤好")
                # x1 = int(l[1] * w)
                # y1 = int(l[2] * h)
                # x2 = int(l[3] * w)
                # y2 = int(l[4] * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)
                #cv2.putText(image, index_2_classes[l[0]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(image, "%s,%s,%s"%(truncation_ratio, overlap_ratio, iou_ratio), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, "%s" %(iou_ratio), (x1, y1),
                #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite("step{}_{}.jpg".format(step, i), image)
            cv2.imshow('show', image)
            cv2.waitKey(1)
        # only one batch
        #break
