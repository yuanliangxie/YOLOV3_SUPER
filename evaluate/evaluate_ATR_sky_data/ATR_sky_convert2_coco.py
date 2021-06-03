import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#                           "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#                           "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#                           "motorbike": 14, "person": 15, "pottedplant": 16,
#                           "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

PRE_DEFINE_CATEGORIES = {"car": 0}


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
		filename = str(filename)
		return filename
	except:
		raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(anno_dirs, json_file):
	json_dict = {"images":[], "type": "instances", "annotations": [],
				 "categories": []}
	categories = PRE_DEFINE_CATEGORIES
	bnd_id = START_BOUNDING_BOX_ID
	for anno_dir in os.listdir(anno_dirs):
		anno_dir_path = os.path.join(anno_dirs, anno_dir)
		for xml_name in os.listdir(anno_dir_path):
			xml_file_path = os.path.join(anno_dir_path, xml_name)
			print("Processing %s"%(xml_file_path))
			tree = ET.parse(xml_file_path)
			root = tree.getroot()

			filename = get_and_check(root, 'filename', 1).text
			filename = anno_dir + '_' + filename

			## The filename must be a number
			image_id = get_filename_as_int(filename)
			size = get_and_check(root, 'size', 1)
			width = int(get_and_check(size, 'width', 1).text)
			height = int(get_and_check(size, 'height', 1).text)
			image = {'file_name': filename, 'height': height, 'width': width,
					 'id':image_id}
			json_dict['images'].append(image)
			## Cruuently we do not support segmentation
			#  segmented = get_and_check(root, 'segmented', 1).text
			#  assert segmented == '0'
			for obj in get(root, 'object'):
				difficult = bool(int(get_and_check(obj, 'difficult', 1).text))
				if difficult:
					print(difficult)
					print('difficult')
					continue
				category = 'car' #这里不用读取，直接固定为car
				#assert category in categories
				category_id = categories[category]
				bndbox = get_and_check(obj, 'bndbox', 1)
				xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
				ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
				xmax = int(get_and_check(bndbox, 'xmax', 1).text)
				ymax = int(get_and_check(bndbox, 'ymax', 1).text)

				assert(xmax > xmin)
				assert(ymax > ymin)
				o_width = abs(xmax - xmin)
				o_height = abs(ymax - ymin)
				ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
					image_id, 'bbox':[xmin, ymin, o_width, o_height],
					   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
					   'segmentation': []}
				json_dict['annotations'].append(ann)
				bnd_id = bnd_id + 1

		for cate, cid in categories.items():
			cat = {'supercategory': 'none', 'id': cid, 'name': cate}
			json_dict['categories'].append(cat)
		json_fp = open(json_file, 'w')
		json_str = json.dumps(json_dict)
		json_fp.write(json_str)
		json_fp.close()


if __name__ == '__main__':
	anno_dirs = "/media/xyl/6418a039-786d-4cd8-b0bb-1ed36a649668/Datasets/sky_challenge_competition/Annotation"
	convert(anno_dirs, "ATR_sky_test.json")