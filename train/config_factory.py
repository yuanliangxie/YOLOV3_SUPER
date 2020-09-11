#U-Detrac的配置文件
from train.Detrac_data_preprocess.params_init_Detrac import TRAINING_PARAMS as Detrac_config_train
from train.Detrac_data_preprocess.params_init_Detrac import Eval as Detrac_config_eval

#VOC的配置文件
from train.Voc_data_preprocess.params_init_voc import TRAINING_PARAMS as Voc_config_train
from train.Voc_data_preprocess.params_init_voc import Eval as Voc_config_eval

#VOC-poly-yolo的配置文件
from train.Voc_data_preprocess.params_init_voc_poly_yolo import TRAINING_PARAMS as Voc_config_train_poly_yolo
from train.Voc_data_preprocess.params_init_voc_poly_yolo import Eval as Voc_config_eval_poly_yolo

_config_factory = {
	"VOC":[Voc_config_train, Voc_config_eval],
	"U-DETRAC":[Detrac_config_train, Detrac_config_eval],
	"VOC_poly_yolo":[Voc_config_train_poly_yolo, Voc_config_eval_poly_yolo]
}

def get_config(config_char):
	return _config_factory[config_char]