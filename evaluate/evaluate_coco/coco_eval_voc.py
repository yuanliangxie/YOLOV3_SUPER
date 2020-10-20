import sys
import os
# MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
sys.path.append("../../../YOLOV3_SUPER")
#from models.model.model_yolov3_baseline import yolov3
from models.model.poly_yolo import yolov3
#from models.model.model_centernet_resnet import centernet_18 as yolov3
from evaluate.evaluate_coco.coco_evaluater import coco_evaluater
#from evaluate.evaluate_coco.yolov3_config_voc_test import TEST as config
from evaluate.evaluate_coco.poly_yolo_config_voc_test import TEST as config
#from evaluate.evaluate_coco.centernet_config_voc_test import TEST as config
from utils.utils_select_device import select_device
import torch
import shutil
os.environ["CUDA_VISIBLE_DEVICES"]='0'

class COCOAPI_evaler(object):
    def __init__(self,
                 gpu_id=0,
                 img_size=config['TEST_IMG_SIZE'],
                 visiual = False,
                 ):
        self.img_size = img_size
        self.__num_class = config['DATA']["NUM"]
        self.__conf_threshold = config["CONF_THRESH"]
        self.__nms_threshold = config["NMS_THRESH"]
        self.__device = select_device(gpu_id)

        self.__visiual = visiual
        self.__classes = config['DATA']["CLASSES"]

        self.__model = yolov3(config)

        # Set data parallel
        #self.__model = nn.DataParallel(self.__model)
        self.__model = self.__model.to(self.__device)

        self.__load_model_weights()

    def __load_model_weights(self):
        if config["pretrain_snapshot"]:
            print("loading weight file from : {}".format(config["pretrain_snapshot"]))
            state_dict = torch.load(config["pretrain_snapshot"])["state_dict"]
            self.__model.load_state_dict(state_dict)
        else:
            print("missing pretrain_snapshot!!!")

        print("loading weight file is done")


    def eval_voc(self):

        print('*' * 20 + "Validate" + '*' * 20)

        with torch.no_grad():
            coco_evaluater(self.__model, config, visiual=self.__visiual).eval()

def pre_process(data_dir_path):
    if not os.path.isdir(data_dir_path):
        return None
    for dir in os.listdir(data_dir_path):
        dir_path = os.path.join(data_dir_path, dir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
def remove_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--visiual', type=bool, default=True, help="get det image in ./data/results")
    opt = parser.parse_args()
    pre_process('./data')
    remove_file('./pred_result.json')
    COCOAPI_evaler(gpu_id=opt.gpu_id,
            visiual=opt.visiual).eval_voc()
