import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from imgaug import augmenters as iaa


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def jaccard_numpy_sample(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A  = A ∩ B / area(A)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    return inter / area_a # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):#TODO：这里出了问题！
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

    def add(self, transform):
        self.transforms.append(transform)


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        x_c = boxes[:, 0] * width
        w = boxes[:, 2] * width
        y_c = boxes[:, 1] * height
        h = boxes[:, 3] * height
        #xywh tran_to x1y1x2y2
        # boxes[:, 0] = x_c - w / 2
        # boxes[:, 1] = y_c - h / 2
        # boxes[:, 2] = x_c + w / 2
        # boxes[:, 3] = y_c + h / 2
        boxes = np.stack((x_c, y_c, w, h), axis=1)
        return image, boxes, labels

class xywh_to_xyxy(object):
    def __call__(self, image, boxes=None, labels=None):
        x_c = boxes[:, 0].copy()#当不做运算时,x_c= boxes[:,0],出现问题，x_c只是boxes[:,0]的引用而已，容易出错，因为下面的代码出现了这样的错误
        y_c = boxes[:, 1].copy()
        w = boxes[:, 2]
        h = boxes[:, 3]

        boxes[:, 0] = x_c - w / 2
        boxes[:, 1] = y_c - h / 2
        boxes[:, 2] = x_c + w / 2
        boxes[:, 3] = y_c + h / 2
        return image, boxes, labels

class xyxy_to_xywh(object):
    def __call__(self, image, boxes=None, labels=None):
        x1 = boxes[:, 0].copy()
        y1 = boxes[:, 1].copy()
        x2 = boxes[:, 2].copy()
        y2 = boxes[:, 3].copy()
        boxes[:, 0] = (x1 + x2) / 2
        boxes[:, 1] = (y1 + y2) / 2
        boxes[:, 2] = x2 - x1
        boxes[:, 3] = y2 - y1
        return image, boxes, labels



class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class ResizeImage_multi_scale(object):
    def __init__(self, batch_size=16, interpolation=cv2.INTER_AREA, mean=(104, 117, 123)):
        self.interpolation = interpolation
        self.image_size = {1: (320, 320), 2: (352, 352), 3: (384, 384), 4: (416, 416), 5: (448, 448), 6: (480, 480),
                           7: (544, 544), 8: (576, 576), 9: (608, 608), 10: (640, 640)
                           }
        self.batch_size_oral = batch_size
        self.batch_size = batch_size
        self.new_size = self.choose_image_size(index=None)
        self.mean = mean

    def __call__(self, image, boxes=None, labels=None):
        if self.batch_size == 0:
            multiplier = int(64*10/self.batch_size_oral)#原论文中batchsize=64,每次经过10次iter就会变换一次尺度
            self.batch_size = self.batch_size_oral * multiplier #这是为了防止当batch_size不是8的倍数时,也能进行多尺度训练
            self.new_size = self.choose_image_size(index=None)
        image, boxes = self.cv2_letterbox_image(image, boxes, self.new_size)
        self.batch_size = self.batch_size - 1

        #将过小的boxes过滤掉！
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        choose = area > 100
        boxes = boxes[choose]
        labels = labels[choose]

        return image, boxes, labels

    def cv2_letterbox_image(self, image, boxes, expected_size):
        """
        使图片固定aspect ratio，并且更新boxes
        :param image:
        :param boxes: shape = [:,4], format:xyxy
        :param expected_size:(,)
        :return:
        """
        ih, iw = image.shape[0:2]
        ew, eh = expected_size
        scale = min(eh / ih, ew / iw)
        boxes = boxes * scale
        nh = int(ih * scale)
        nw = int(iw * scale)
        image = cv2.resize(image, (nw, nh), interpolation=self.interpolation)
        top = (eh - nh) // 2
        bottom = eh - nh - top
        left = (ew - nw) // 2
        right = ew - nw - left
        boxes[:, [0,2]] = boxes[:, [0,2]] + left
        boxes[:, [1,3]] = boxes[:, [1,3]] + top#boxes里的坐标不会出界，因为此图片扩充，框永远在图片中
        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.mean)
        return new_img, boxes

    def choose_image_size(self, index=None):
        if index is None:
            index = random.randint(1, len(self.image_size)+1)
        return tuple(self.image_size[index])

class ResizeImage_single_scale(object):
    def __init__(self, new_size, interpolation=cv2.INTER_AREA, mean = (104, 117, 123)):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation
        self.mean = mean

    def cv2_letterbox_image(self, image, boxes, expected_size):
        """
        使图片固定aspect ratio，并且更新boxes
        :param image:
        :param boxes: shape = [:,4], format:xyxy
        :param expected_size:(,)
        :return:
        """
        ih, iw = image.shape[0:2]
        ew, eh = expected_size
        scale = min(eh / ih, ew / iw)
        boxes = boxes * scale
        nh = int(ih * scale)
        nw = int(iw * scale)
        image = cv2.resize(image, (nw, nh), interpolation=self.interpolation)
        top = (eh - nh) // 2
        bottom = eh - nh - top
        left = (ew - nw) // 2
        right = ew - nw - left
        boxes[:, [0,2]] = boxes[:, [0,2]] + left
        boxes[:, [1,3]] = boxes[:, [1,3]] + top#boxes里的坐标不会出界，因为此图片扩充，框永远在图片中
        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.mean)
        return new_img, boxes

    def __call__(self, image, boxes=None, labels=None):
        image, boxes = self.cv2_letterbox_image(image, boxes, self.new_size)

        #将过小的boxes过滤掉！
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        choose = area > 100
        boxes = boxes[choose]
        labels = labels[choose]

        return image, boxes, labels



class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __init__(self, is_debug=False):
        self.is_debug = is_debug

    def __call__(self, image, boxes=None, labels=None):
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))  # 将（416, 416, 3）换成（3, 416, 416）
            #image = image.astype(np.float32)
            return torch.from_numpy(image).type(torch.FloatTensor),\
                    torch.from_numpy(boxes).type(torch.FloatTensor),\
                     torch.from_numpy(labels).type(torch.FloatTensor)
        else:
            image = image.astype(np.uint8)
            return torch.from_numpy(image),\
                   torch.from_numpy(boxes).type(torch.FloatTensor),\
                   torch.from_numpy(labels).type(torch.FloatTensor)



class RandomSample_for_all_scales(object):#详见LFFD论文数据增强方法
    def __init__(self, mean):
        #self.continuous_face_scale=[[10, 15], [15, 20], [20, 40], [40, 70], [70, 110], [110, 250], [250, 400], [400, 560]]
        self.continuous_face_scale=[[15, 45], [45, 75], [75, 135], [135, 260]]
        #TODO:delete
        #self.continuous_face_scale=[[15, 45]]
        self.fix_image_size = 640
        self.mean = mean


    def __call__(self, image, boxes=None, labels=None):

        bs = boxes.shape[0]
        random_choice = random.randint(0, bs)
        choose_boxes_xyxy = boxes[random_choice]
        choose_boxes_center_xy = (choose_boxes_xyxy[2:] + choose_boxes_xyxy[:2])/2
        max_side = np.max(choose_boxes_xyxy[2:]-choose_boxes_xyxy[:2])
        random_face_scale = self.continuous_face_scale[random.randint(0, len(self.continuous_face_scale))]
        random_resize_value = random.uniform(random_face_scale[0], random_face_scale[1])
        alpha = random_resize_value/max_side
        self.crop_image_size = int(self.fix_image_size/alpha)
        padding_image = np.ones((self.crop_image_size, self.crop_image_size, 3), dtype=np.uint8)

        for i in range(3):
            padding_image[..., i] = padding_image[..., i] * self.mean[i] #进行均值的初始化

        crop_image_size_boxes = np.append(choose_boxes_center_xy - int(self.crop_image_size/2),
                                          choose_boxes_center_xy + int(self.crop_image_size/2))

        non_occlution = jaccard_numpy_sample(boxes, crop_image_size_boxes)
        mask = non_occlution > 0.5
        boxes = boxes[mask==True]
        labels = labels[mask==True]

        zeros_point = choose_boxes_center_xy - int(self.crop_image_size/2)

        crop_image_position = self.calculate_jaccard_position(crop_image_size_boxes,
                                                              np.array([[0, 0, image.shape[1], image.shape[0]]]))
        x1, y1, x2, y2 = crop_image_position[0]

        relative_crop_positon = self.calculate_relative_position(zeros_point, crop_image_position)

        rela_x1, rela_y1, rela_x2, rela_y2 = relative_crop_positon[0]

        min_w = int(min(x2-x1, rela_x2- rela_x1))
        min_h = int(min(y2-y1, rela_y2- rela_y1))

        padding_image[int(rela_y2)-min_h:int(rela_y2), int(rela_x2)-min_w:int(rela_x2), :] = image[int(y2)-min_h:int(y2), int(x2)-min_w:int(x2), :]

        image_position_boxes = self.calculate_jaccard_position(crop_image_size_boxes, boxes)

        boxes = self.calculate_relative_position(zeros_point, image_position_boxes)


        image = cv2.resize(padding_image, (640, 640), interpolation=cv2.INTER_AREA)

        boxes = boxes * 640/padding_image.shape[0]

        #防止boxes越界，好像不防止也不要紧，因为这是当做预测的值的。
        # boxes[:, :2] = np.clip(boxes[:, :2], a_min=0, a_max=np.inf)
        # boxes[:, 2:] = np.clip(boxes[:, 2:], a_min=0, a_max=self.fix_image_size)


        #将过小的boxes过滤掉！
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        choose = area > 225
        boxes = boxes[choose]
        labels = labels[choose]

        return image, boxes, labels


    def calculate_jaccard_position(self, crop_image_size_boxes, boxes):
        left_top = np.maximum(crop_image_size_boxes[:2], boxes[:, :2])
        rignt_bottom = np.minimum(crop_image_size_boxes[2:], boxes[:, 2:])
        return np.concatenate([left_top, rignt_bottom], axis=1)

    def calculate_relative_position(self, zeros_point, jaccard_boxes):
        zeros_point_xyxy = np.append(zeros_point, zeros_point)
        return jaccard_boxes - zeros_point_xyxy



class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean
        """
        Expand函数将图片扩展，并使原始大小图片在扩展的图片中任意放置！
        """

    def __call__(self, image, boxes, labels):
        if random.randint(2):#有一定几率返回原图！
            return image, boxes, labels
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        im, boxes, labels = self.rand_light_noise(im, boxes, labels)
        return im, boxes, labels

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential([
                              sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.2)),
                              sometimes(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5))],
                             random_order=True)


    def __call__(self, image, boxes, labels):
        image = self.seq(image=image)
        return image, boxes, labels

# class Mixup(object):
#     def __init__(self, alpha):
#         self.alpha = alpha
#
#     def __call__(self, image, boxes, labels):



class SSDAugmentation(object):
    def __init__(self, mean=(104, 117, 123)):
        self.augment = Compose([
            #ConvertFromInts(),
            ToAbsoluteCoords(),
            xywh_to_xyxy(),
            #PhotometricDistort(),
            ImageBaseAug(),
        # Expand(mean),            #-----|
        # RandomSampleCrop(),      #-----|-当选用yolov3系列时取消注释这两行
            RandomSample_for_all_scales(mean),#当选用LVD系列
            RandomMirror(),
            #xyxy_to_xywh(),
            #ToPercentCoords(),

        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


