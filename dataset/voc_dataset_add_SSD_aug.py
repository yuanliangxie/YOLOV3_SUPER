import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from dataset import ssd_augmentations



class VOCDataset(Dataset):
    def __init__(self, list_path, labels_path, img_size, is_training, is_debug=False, batch_size=16):
        self.img_files = []
        self.label_files = []
        self.batch_size = batch_size
        # if is_training:
        #     labels_path = '../data/voc/labels'
        # else:
        #     labels_path = '../data/voc/labels_test'
        for path in open(list_path, 'r'): #'vehecal/labels'
            path_split = path.split('/')
            index_jpg = path_split[-1]
            label_path = os.path.join(labels_path, index_jpg.replace('.jpg', '.txt')).strip()
            # label_path = path.replace('VOC2007/JPEGImages', 'labels').replace('VOC2012/JPEGImages', 'labels').replace(
            #  'VOC2007_test/JPEGImages', 'labels_test').replace('.jpg', '.txt').strip()
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                print("no label found. skip it: {}".format(path))
        #print("Total images: {}".format(len(self.img_files)))
        #self.img_size = img_size  # (w, h)
        #self.max_objects = 50#这里的max_object是最大的bbox的数目
        self.is_debug = is_debug

        #  transforms and augmentation
        self.total_transforms = ssd_augmentations.Compose(transforms=[])

        # self.transforms.add(data_transforms.KeepAspect())

        if is_training:
            self.total_transforms.add(ssd_augmentations.SSDAugmentation())
            if img_size is None:
                self.total_transforms.add(ssd_augmentations.ResizeImage_multi_scale(batch_size=self.batch_size))
            else:
                self.total_transforms.add(ssd_augmentations.ResizeImage_single_scale(new_size=img_size))
        else:
            self.total_transforms.add(ssd_augmentations.ToAbsoluteCoords())
            self.total_transforms.add(ssd_augmentations.xywh_to_xyxy())
            self.total_transforms.add(ssd_augmentations.ResizeImage_single_scale(new_size=img_size))
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
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#TODO:加入训练的数据是RGB格式的

        label_path = self.label_files[index].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)#np.loadtxt可以直接加载文本中具有固定格式的文字数据
        else:
            print("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)
        image = img
        box = labels[:, 1:]
        label = labels[:, 0].reshape(-1, 1)

        if self.total_transforms is not None:
            image, box, label = self.total_transforms(image, box, label)

        label = torch.cat((label, box), 1)

        img_ind = img_path[-10:-4]
        # sample = {'image': image, 'label': torch.cat((label, box),1)}
        # sample["image_path"] = img_path[-10:-4]
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
            filled_labels = torch.zeros(max_objects, 5).type(torch.FloatTensor)
            filled_labels[range(max_objects)[:label_len]] = label[i][:]
            label_new.append(filled_labels)
        sample = {'image': torch.stack(image), 'label': torch.stack(label_new), 'img_ind': img_ind, "origin_size": origin_size}
        return sample


#  use for test dataloader
if __name__ == "__main__":
    import sys
    MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
    import train.Voc_data_preprocess.params_init_voc as params_init
    classes_category = params_init.TRAINING_PARAMS["yolo"]["classes_category"]
    vocdataset = VOCDataset("../data/voc/test.txt", "../data/voc/labels_test",
                            None, True, is_debug=False, batch_size=8)
    index_2_classes = vocdataset.index_2_classes(classes_category)
    dataloader = torch.utils.data.DataLoader(vocdataset,
                                             batch_size=8,
                                             shuffle=False, num_workers=0, pin_memory=False, collate_fn=vocdataset.collate_fn)
    print(len(vocdataset))
    from models.bricks.tricks import mix_up
    mix_up_method = mix_up(1)
    for step, sample in enumerate(dataloader):
        #print(step)
        images, labels = sample['image'], sample['label']
        images, labels = mix_up_method.mixup(images, labels)
        for i, (image, label) in enumerate(zip(images, labels)):
            image = image.numpy()*255
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            label = label.numpy()
            h, w = image.shape[:2]
            for l in label:
                if l.sum() == 0:
                    continue
                x1 = int((l[1] - l[3] / 2) * w)
                y1 = int((l[2] - l[4] / 2) * h)
                x2 = int((l[1] + l[3] / 2) * w)
                y2 = int((l[2] + l[4] / 2) * h)
                # x1 = int(l[1] * w)
                # y1 = int(l[2] * h)
                # x2 = int(l[3] * w)
                # y2 = int(l[4] * h)
                image = np.ascontiguousarray(image)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)
                image = cv2.putText(image, index_2_classes[l[0]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite("step{}_{}.jpg".format(step, i), image)
            cv2.imshow('show', image)
            cv2.waitKey(2000)
        # only one batch
        #break
