import sys
import os
# MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
sys.path.append("../../../YOLO_SUPER")
from models.model.model_yolov3_baseline import yolov3
from evaluate.evaluate_detrac_coco_api.coco_evaluater import coco_evaluater
from evaluate.evaluate_detrac_coco_api.yolov3_config_dtrac_test import TEST as config
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

    @staticmethod
    def eval_data_with_result_file():
        print('*' * 20 + "直接通过已生成的预测结果文件进行Validate" + '*' * 20)
        coco_evaluater.eval()

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    # parser.add_argument('--visiual', type=str, default='./data/test', help='data augment flag')
    # parser.add_argument('--eval', action='store_true', default=True, help='data augment flag')
    # parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    # opt = parser.parse_args()
    scrach_run_flag = True
    if scrach_run_flag == False:
        assert os.path.isfile('pred_result.json'), "找不到pred_result.json文件，请将scrach_run_flag设置为True"
        COCOAPI_evaler.eval_data_with_result_file()
    else:
        pre_process('data')
        remove_file('pred_result.json')
        gpu_id = 0
        visiual = True
        COCOAPI_evaler(gpu_id=gpu_id,
                visiual=visiual).eval_voc()
