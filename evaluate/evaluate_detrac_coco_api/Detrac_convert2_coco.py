import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from train.Detrac_data_preprocess.Detrac_data_process import txt_data

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#                           "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#                           "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#                           "motorbike": 14, "person": 15, "pottedplant": 16,
#                           "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
PRE_DEFINE_CATEGORIES = {'car':0, 'bus':1, 'van':2, 'others':3}

def get_xml_list(xml_dir, txt_path ='xml_list.txt'):
    xml_lists = os.listdir(xml_dir)
    f = open(txt_path, 'w')
    for xml_Subpath in xml_lists:
        f.write(xml_Subpath+"\n")
    f.close()


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.split('.')[0].strip()
        #filename = os.path.splitext(filename)[0]
        #filename = int(filename)
        filename = str(filename)
        return filename
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')#这里要与Detrac_data_process的测试集一致
    #list_fp = ["MVI_39401.xml", "MVI_40711.xml", "MVI_40712.xml"]
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)

        if not os.path.isfile(xml_f):
            raise NotImplementedError('paths is not found in %s'%(xml_f))

        tree = ET.parse(xml_f)
        root = tree.getroot()
        video_name = root.attrib['name']
        ignored_region = root.find('ignored_region')
        ignored_region_list = []
        for box in ignored_region:  # 将忽略区域的坐标写入f_ignore中
            x1y1wh = list(box.attrib.values())
            x1y1wh = [float(str) for str in x1y1wh]  # TODO:需要计算与掩模的IOU比例,增加一个计算与mask相交比例的ratio！
            ignored_region_list.append(x1y1wh)  # 收集掩模！
        if len(ignored_region_list) == 0:
            ignored_region_array = None
        else:
            ignored_region_array = np.stack(ignored_region_list, 0)
        #test_image_dir = os.path.join(test_image_dir, video_name)
        frame_all = root.findall('frame')
        image_height = 540.0
        image_width = 960.0
        for frame in frame_all:
            frame_number = frame.attrib['num']
            target_list = frame.find('target_list')
            image_id_path = video_name + frame_number.zfill(5)
            image = {'file_name': image_id_path, 'height': image_height, 'width': image_width,
                     'id': image_id_path}
            json_dict['images'].append(image)
            for is_object_path, ann in get_detrac_info(target_list, ignored_region_array, categories, image_id_path, bnd_id):
                if is_object_path:
                    ann['id'] = bnd_id
                    print(bnd_id)
                    bnd_id  = bnd_id +1
                    json_dict['annotations'].append(ann)

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    #list_fp.close()



def get_detrac_info(target_list, ignored_region_array, category_dict, image_id, bnd_id,
                    filter_truncation_ratio=0.95, filter_occlusion_ratio=0.95, filter_mask_iouratio=0.4):
    is_object_path = False
    image_height = 540.0
    image_width = 960.0
    box_list = []
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
        if class_name not in category_dict.keys():#判断是不是所要检测的目标标签
            continue

############################################################################################################################
        is_object_path = True#只要循环能下来就说明有物体不过还没有经过mask_filter
        class_index = category_dict[class_name]
        class_index_list.append([class_index])
        box_list.append(x1y1wh)

#################################################mask_occlusiont_filter##############################################
    if len(class_index_list) == 0:
        return is_object_path, None
    class_index_array = np.stack(class_index_list, 0)
    box_array = np.stack(box_list, 0)
    if type(ignored_region_array) is not np.ndarray:
        class_index_array = class_index_array
        box_array = box_array
    else:
        occlusion_iou = txt_data.mask_iou_ratio(box_array = box_array, ignored_region_array=ignored_region_array)
        iou_ratio_array = np.sum(occlusion_iou, 1)#这里也可用max，考虑到一辆车在边界上可能与多块掩模覆盖相交，所以我们计算总和！
        iou_ratio_array = np.expand_dims(iou_ratio_array, axis=1)
        occlusion_mask = iou_ratio_array[:, 0] < filter_mask_iouratio
        if not occlusion_mask.any():  # 经过掩码过滤后全部框都没了，直接返回False
            is_object_path = False
            return is_object_path, None
        class_index_array = class_index_array[occlusion_mask, :]
        box_array = box_array[occlusion_mask, :]
    attributes_array = np.concatenate([class_index_array, box_array], 1)

#########################################################写入有效的bboxes和属性#############################################
    for index, (class_index, x1, y1, w, h) in enumerate(attributes_array):
        x1 = int(x1)
        y1 = int(y1)
        o_width = int(w)
        o_height = int(h)
        ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [x1, y1, o_width, o_height],
               'category_id': class_index, 'id': 0, 'ignore': 0,
               'segmentation': []}
        yield is_object_path, ann


if __name__ == '__main__':
    # if len(sys.argv) <= 1:
    #     print('3 auguments are need.')
    #     print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
    #     exit(1)

    xml_dir = "/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/UA-DETRAC/DETRAC-Test-Annotations-XML"
    #test_image_dir ="/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/UA-DETRAC/DETRAC-test-data"
    xml_list_txt = "Detrac_xml_list.txt"
    get_xml_list(xml_dir, xml_list_txt)
    convert(xml_list_txt, xml_dir, "Detrac_test.json")