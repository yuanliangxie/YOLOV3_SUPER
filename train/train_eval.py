# from evaluate_detrac_coco_api.coco_evaluater import coco_evaluater
# from train_process.Detrac_data_preprocess.params_init_Detrac import Eval as config
# from train_process.Detrac_data_preprocess.params_init_Detrac import TRAINING_PARAMS as config_train

# from evaluate.evaluate_coco.coco_evaluater import coco_evaluater
# from train.Voc_data_preprocess.params_init_voc import Eval as config
# from train.Voc_data_preprocess.params_init_voc import TRAINING_PARAMS as config_train


from utils.utils_select_device import select_device
import torch
#import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# config['DATA'] = {}
# config['DATA']["CLASSES"] = config_train['yolo']['classes_category']
# config['DATA']["NUM"] = config_train['yolo']['classes']

class train_evaler(object):
    def __init__(self,
                 model,
                 logger,
                 config_eval,
                 coco_evaluater,
                 visiual = True,
                 ):
        self.config = config_eval
        self.img_size = (self.config['TEST_IMG_SIZE'], self.config['TEST_IMG_SIZE'])
        self.logger = logger
        self.__num_class = self.config['DATA']["NUM"]
        self.__conf_threshold = self.config["CONF_THRESH"]
        self.__nms_threshold = self.config["NMS_THRESH"]
        self.__visiual = visiual
        self.__classes = self.config['DATA']["CLASSES"]
        self.model = model
        self.coco_evaluater = coco_evaluater

        # Set data parallel
        #self.__model = nn.DataParallel(self.__model)

    def eval_voc(self):
        print('*' * 20 + "Validate" + '*' * 20)

        with torch.no_grad():
            state = self.coco_evaluater(self.model, self.config, visiual=self.__visiual).eval()
            mAP = round(state[1], 4)
            self.logger.append('mAP:%g' % (mAP))
        return mAP
