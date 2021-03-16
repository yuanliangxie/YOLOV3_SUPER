import sys
import os
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.path.append("../../../YOLOV3_SUPER")
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import numpy as np

class txt_data(object):
    def __init__(self, data_path):
        """

        :param voc_data_path: the dirpath of the vocdata
        :param classes: 该数据集的全部种类
        because the class metheod has implement the Any class of training labels in voc
        :TODO: the class can func only in voc, but the method has generalization ability,so it can expand to other datasets.
        :TODO：如果想要使两个类别合并为一类，同样可以修改class_2_index,自输入额外的index，使两个不同类别对应的index一样即可。
        """
        super(txt_data, self).__init__()
        self.Detrac_data_path = data_path
        self.train_anno_path_dir = os.path.join(self.Detrac_data_path, "DETRAC-Train-Annotations-XML")
        self.test_anno_path_dir = os.path.join(self.Detrac_data_path, "DETRAC-Test-Annotations-XML")
        self.train_image_dir = os.path.join(self.Detrac_data_path, "DETRAC-train-data")
        self.test_image_dir = os.path.join(self.Detrac_data_path, "DETRAC-test-data")
        self.data_dir = "../../data"
        self.dir_path = "../../data/detrac"
        self.labels_path = self.dir_path + '/' + 'labels'
        self.labels_test_path = self.dir_path + '/' + 'labels_test'

        self.classes = ['car', 'bus', 'van', 'others']
        self.class_2_index = self._class_2_index()
        self.clear_make_dir()

    def _class_2_index(self):#todo:如果后面要进行类的合并，则在这里修改就可以了
        #index = [i for i in range(len(self.classes))] #这里是分为4类
        index = [0 for i in range(len(self.classes))] #将所有种类都合并为0，一类car类
        class_2_index = dict(zip(self.classes, index))
        return class_2_index

    def clear_make_dir(self):
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)

        if not os.path.isdir(self.labels_path):
            os.mkdir(self.labels_path)
        else:
            shutil.rmtree(self.labels_path)
            os.mkdir(self.labels_path)
            print("\n对"+self.labels_path+"中的文件进行删除！\n")

        if not os.path.isdir(self.labels_test_path):
            os.mkdir(self.labels_test_path)
        else:
            shutil.rmtree(self.labels_test_path)
            os.mkdir(self.labels_test_path)
            print("\n对"+self.labels_test_path+"中的文件进行删除！\n")

    def _write_anno_detrac_txt(self, target_list, txt_path, ignored_region_array,
                               filter_truncation_ratio=0.95, filter_occlusion_ratio=0.95, filter_mask_iouratio=0.4):
        is_object_path = False
        f_txt = open(txt_path, 'w')
        image_height = 540.0
        image_width = 960.0
        box_list = []
        truncation_ratio_list = []
        overlap_ratio_list = []
        class_index_list = []
        for target in target_list:
            is_occlusion = False
            overlap_ratio = 0
            for obj_anno in target:

                if obj_anno.tag == "box":
                    box_dict = obj_anno.attrib
                    x1y1wh = list(box_dict.values())
                    x1y1wh = [float(coor) for coor in x1y1wh]

                elif obj_anno.tag == "attribute":
                    attribute = obj_anno.attrib
                    truncation_ratio = float(attribute['truncation_ratio'])

                elif obj_anno.tag == "occlusion":

                    occlusion = obj_anno #这里有重叠的信息，如果以后要根据重叠的程度去掉一些标签，则可以在这里修改！
                    region_overlap = occlusion[0].attrib
                    is_occlusion = True

##################################################过滤遮挡的物体########################################################
            if is_occlusion:
                w, h = float(box_dict['width']), float(box_dict['height'])
                w_o, h_o = float(region_overlap['width']), float(region_overlap['height'])
                overlap_ratio = (w_o * h_o)/(w * h)



            if overlap_ratio > filter_occlusion_ratio:
                print("occlusion_filter")
                continue

##################################################过滤被截断的物体######################################################
            if truncation_ratio > filter_truncation_ratio:
                print("truncation_filter")
                continue

##################################################过滤不需要检测的车辆类别######################################################
            class_name = attribute['vehicle_type']
            if class_name not in self.class_2_index.keys():#判断是不是所要检测的目标标签
                continue

############################################################################################################################
            is_object_path = True#只要循环能下来就说明有物体不过还没有经过mask_filter
            class_index = self.class_2_index[class_name]
            class_index_list.append([class_index])
            box_list.append(x1y1wh)
            truncation_ratio_list.append([truncation_ratio])
            overlap_ratio_list.append([overlap_ratio])

#################################################mask_occlusiont_filter##############################################
        if len(class_index_list) == 0:
            return is_object_path
        class_index_array = np.stack(class_index_list, 0)
        truncation_ratio_array = np.stack(truncation_ratio_list, 0)
        overlap_ratio_array = np.stack(overlap_ratio_list, 0)
        box_array = np.stack(box_list, 0)
        if type(ignored_region_array) is not np.ndarray:
            class_index_array = class_index_array
            box_array = box_array
            truncation_ratio_array = truncation_ratio_array
            overlap_ratio_array = overlap_ratio_array
            iou_ratio_array = np.zeros_like(overlap_ratio_array)
        else:
            occlusion_iou = self.mask_iou_ratio(box_array, ignored_region_array)
            iou_ratio_array = np.sum(occlusion_iou, 1)#这里也可用max，考虑到一辆车在边界上可能与多块掩模覆盖相交，所以我们计算总和！
            iou_ratio_array = np.expand_dims(iou_ratio_array, axis=1)
            occlusion_mask = iou_ratio_array[:, 0] < filter_mask_iouratio
            if not occlusion_mask.any(): #经过掩码过滤后全部框都没了，直接返回False
                is_object_path = False
                return is_object_path
            # occlusion_mask = np.zeros(occlusion_iou.shape[0], dtype=np.bool)
            # occlusion_bool = occlusion_iou > 0.5
            # for i in range(occlusion_iou.shape[0]):
            #     if occlusion_bool[i, :].any():
            #         occlusion_mask[i] = True
            class_index_array = class_index_array[occlusion_mask, :]#TODO;
            box_array = box_array[occlusion_mask, :]
            iou_ratio_array = iou_ratio_array[occlusion_mask, :]
            truncation_ratio_array = truncation_ratio_array[occlusion_mask, :]
            overlap_ratio_array = overlap_ratio_array[occlusion_mask, :]
        attributes_array = np.concatenate([class_index_array, box_array, truncation_ratio_array, overlap_ratio_array, iou_ratio_array], 1)

#########################################################写入有效的bboxes和属性#############################################
        for index, (class_index, x1, y1, w, h, truncation_ratio, overlap_ratio, iou_ratio) in enumerate(attributes_array):
            x2 = x1 + w
            y2 = y1 + h
            x_c = (x1 + x2) / 2 / image_width
            y_c = (y1 + y2) / 2 / image_height
            w = w / image_width
            h = h / image_height
            str_list = [class_index, x_c, y_c, w, h, truncation_ratio, overlap_ratio, iou_ratio]#加入截断率和重叠率
            str_write = ' '.join(['%.3f' % i for i in str_list])
            f_txt.write(str_write + '\n')
        f_txt.close()
        return is_object_path

    @staticmethod
    def mask_iou_ratio(box_array, ignored_region_array):
        """

        :param box_array:
        :param ignored_region_array:
        :return:
        """
        boxes1 = box_array
        boxes2 = ignored_region_array

        b1x1 = boxes1[:, 0].reshape(-1, 1, 1)
        b1y1 = boxes1[:, 1].reshape(-1, 1, 1)
        b1x2 = (boxes1[:, 0] + boxes1[:, 2]).reshape(-1, 1, 1)
        b1y2 = (boxes1[:, 1] + boxes1[:, 3]).reshape(-1, 1, 1)

        b2x1 = boxes2[:, 0].reshape(-1, 1)
        b2y1 = boxes2[:, 1].reshape(-1, 1)
        b2x2 = (boxes2[:, 0] + boxes2[:, 2]).reshape(-1, 1)
        b2y2 = (boxes2[:, 1] + boxes2[:, 3]).reshape(-1, 1)

        x1 = np.maximum(b1x1, b2x1)
        y1 = np.maximum(b1y1, b2y1)
        x2 = np.minimum(b1x2, b2x2)
        y2 = np.minimum(b1y2, b2y2)

        dx = np.clip(x2-x1, 0, np.inf)
        dy = np.clip(y2-y1, 0, np.inf)

        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        return np.squeeze(intersections/areas1, axis=-1)



    def generate_train_txt(self):
        f = open(os.path.join(self.dir_path, 'train.txt'), 'w')
        f_ignore = open(os.path .join(self.dir_path, 'train_ignore_region.txt'), 'w')
        detrac_train_anno_paths = os.listdir(self.train_anno_path_dir)
        for detrac_train_anno_path in tqdm(detrac_train_anno_paths):
            anno_path = os.path.join(self.train_anno_path_dir, detrac_train_anno_path)
            tree = ET.parse(anno_path)
            root = tree.getroot()
            video_name = root.attrib['name']
            f_ignore.write(video_name+" ")
            ignored_region = root.find('ignored_region')
            ignored_region_list = []
            for box in ignored_region: #将忽略区域的坐标写入f_ignore中
                x1y1wh = list(box.attrib.values())
                str = " ".join(x1y1wh)
                f_ignore.write(str+" ")
                x1y1wh = [float(str) for str in x1y1wh]#TODO:需要计算与掩模的IOU比例,增加一个计算与mask相交比例的ratio！
                ignored_region_list.append(x1y1wh)#收集掩模！
            if len(ignored_region_list) == 0:
                ignored_region_array = None
            else:
                ignored_region_array = np.stack(ignored_region_list, 0)
            f_ignore.write("\n")
            train_image_dir = os.path.join(self.train_image_dir, video_name)
            frame_all = root.findall('frame')
            for frame in frame_all:
                frame_number = frame.attrib['num']
                frame_max_object = frame.attrib['density']
                target_list = frame.find('target_list')
                path = video_name+frame_number.zfill(5)
                txt_path = os.path.join(self.labels_path, path + '.txt')
                is_object_path = self._write_anno_detrac_txt(target_list, txt_path, ignored_region_array)
                if is_object_path:
                    f.write(os.path.join(train_image_dir, "img"+frame_number.zfill(5) + '.jpg' + '\n'))
        f.close()
        f_ignore.close()

    def generate_test_txt(self):
        f = open(os.path.join(self.dir_path, 'test.txt'), 'w')
        f_ignore = open(os.path.join(self.dir_path, 'test_ignore_region.txt'), 'w')
        #detrac_test_anno_paths = os.listdir(self.test_anno_path_dir)#这是全部测试数据集，但是太大了，所以进行选择缩小
        detrac_test_anno_paths = ["MVI_39401.xml", "MVI_40711.xml", "MVI_40712.xml"]#TODO:这里有两个选项生成不同的测试集!
        #["MVI_39401.xml", "MVI_40711.xml", "MVI_40712.xml"]
        for detrac_test_anno_path in tqdm(detrac_test_anno_paths):
            anno_path = os.path.join(self.test_anno_path_dir, detrac_test_anno_path)
            tree = ET.parse(anno_path)
            root = tree.getroot()
            video_name = root.attrib['name']
            f_ignore.write(video_name + " ")
            ignored_region = root.find('ignored_region')
            ignored_region_list = []
            for box in ignored_region:  # 将忽略区域的坐标写入f_ignore中
                x1y1wh = list(box.attrib.values())
                str = " ".join(x1y1wh)
                f_ignore.write(str + " ")
                x1y1wh = [float(str) for str in x1y1wh]  # TODO:需要计算与掩模的IOU比例,增加一个计算与mask相交比例的ratio！
                ignored_region_list.append(x1y1wh)  # 收集掩模！
            if len(ignored_region_list) == 0:
                ignored_region_array = None
            else:
                ignored_region_array = np.stack(ignored_region_list, 0)
            f_ignore.write("\n")
            test_image_dir = os.path.join(self.test_image_dir, video_name)
            frame_all = root.findall('frame')
            for frame in frame_all:
                frame_number = frame.attrib['num']
                frame_max_object = frame.attrib['density']
                target_list = frame.find('target_list')
                path = video_name + frame_number.zfill(5)
                txt_path = os.path.join(self.labels_test_path, path + '.txt')
                is_object_path = self._write_anno_detrac_txt(target_list, txt_path, ignored_region_array)
                if is_object_path:
                    f.write(os.path.join(test_image_dir, "img" + frame_number.zfill(5) + '.jpg' + '\n'))
        f.close()
        f_ignore.close()


    def print_class_2_index(self):
        print("label_dict:" + str(self.class_2_index))

def main():
    #模型想要什么，数据就需要生产什么
    import train.Detrac_data_preprocess.params_init_Detrac as params_init
    data_path = params_init.TRAINING_PARAMS['data_path']
    generate_txt = txt_data(data_path)
    generate_txt.generate_train_txt()
    generate_txt.generate_test_txt()
    print('\n')
    generate_txt.print_class_2_index()


if __name__ == '__main__':
    main()