#用类重构的生成数据地址和标签的程序
import sys
import os
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.path.append("../../../YOLOV3_SUPER")
import train.Research_data_preprocess.params_inits as params_init
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

class txt_data(object):
	def __init__(self, VOC_format_data_path, classes, data_name="research_data"):
		"""

		:param VOC_format_data_path: the dirpath of the vocdata
		:param classes: the classes what you want to train ,just set what you want,
		because the class metheod has implement the Any class of training labels in voc
		:TODO: the class can func only in voc, but the method has generalization ability,so it can expand to other datasets.
		:TODO：如果想要使两个类别合并为一类，同样可以修改class_2_index,自输入额外的index，使两个不同类别对应的index一样即可。
		"""
		super(txt_data, self).__init__()
		self.researchData = VOC_format_data_path
		self.researchData_test_path = self.researchData + '/ImageSets/Main/test.txt'

		self.data_dir = "../../data"
		self.dir_path = "../../data/"+data_name
		self.labels_path = self.dir_path + '/' + 'labels'
		self.labels_test_path = self.dir_path + '/' + 'labels_test'

		self.classes = classes
		self.class_2_index = self._class_2_index()
		self.clear_make_dir()

	def _class_2_index(self):#todo:如果后面要进行类的合并，则在这里修改就可以了
		index = [i for i in range(len(self.classes))]
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
			print("\n对voc/labels中的文件进行删除！\n")

		if not os.path.isdir(self.labels_test_path):
			os.mkdir(self.labels_test_path)
		else:
			shutil.rmtree(self.labels_test_path)
			os.mkdir(self.labels_test_path)
			print("\n对voc/label_test中的文件进行删除！\n")

	def _write_anno_txt(self, anno_path, txt_path, is_train):
		#is_object_path是判断这个anno_path里面是否含有检测目标类别，有则返回true,无则返回false
		is_object_path = False
		tree = ET.parse(anno_path)
		image_size = tree.find('size')
		image_height = float(image_size.find('height').text)
		image_width = float(image_size.find('width').text)
		objects = tree.findall('object')
		f_txt = open(txt_path, 'w')
		for obj in objects:
			class_name = obj.find('name').text
			if class_name not in self.class_2_index.keys():#判断是不是所要检测的目标标签
				continue

			# class_is_diff = bool(int(obj.find('difficult').text))
			# if is_train:  # 如果需要困难的标签，则将其注释！
			# 	if class_is_diff:
			# 		continue
			is_object_path = True
			box = obj.find('bndbox')
			x1 = float(box.find('xmin').text)
			y1 = float(box.find('ymin').text)
			x2 = float(box.find('xmax').text)
			y2 = float(box.find('ymax').text)
			x_c = (x2 + x1) / 2 / image_width
			y_c = (y1 + y2) / 2 / image_height
			w = (x2 - x1) / image_width
			h = (y2 - y1) / image_height
			# if x1/image_width > 1 or y1/image_height > 1 or x2/image_width >1 or y2/image_height>1:
			#     print("ohohoh!")
			str_list = [self.class_2_index[class_name], x_c, y_c, w, h]
			str_write = ' '.join(['%.3f' % i for i in str_list])
			f_txt.write(str_write + '\n')
		f_txt.close()
		return is_object_path

	def generate_test_txt(self):
		with open(os.path.join(self.dir_path, 'test.txt'), 'w') as f:
			with open(self.researchData_test_path, 'r') as f_test:
				for path in tqdm(f_test.readlines()):
					path = path.strip()
					Anno_path = os.path.join(self.researchData, 'Annotations', path + '.xml')
					txt_path = os.path.join(self.labels_test_path, path + '.txt')
					is_object_path = self._write_anno_txt(Anno_path, txt_path, is_train=False)
					if is_object_path:
						f.write(os.path.join(self.researchData, 'JPEGImages', path + '.jpg' + '\n'))

	def print_class_2_index(self):
		print("label_dict:" + str(self.class_2_index))

def main():
	classes_category = params_init.TRAINING_PARAMS["model"]["classes_category"]
	voc_data_path = params_init.TRAINING_PARAMS['data_path']
	generate_txt = txt_data(voc_data_path, classes_category, "research_11_50_data")
	#generate_txt.generate_train_txt()
	generate_txt.generate_test_txt()
	print('\n')
	generate_txt.print_class_2_index()


if __name__ == '__main__':
	main()