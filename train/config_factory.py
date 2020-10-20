#U-Detrac的配置文件
from train.Detrac_data_preprocess.params_init_Detrac import TRAINING_PARAMS as Detrac_config_train
from train.Detrac_data_preprocess.params_init_Detrac import Eval as Detrac_config_eval

from train.Detrac_data_preprocess.params_init_Detrac_centernet import TRAINING_PARAMS as Detrac_config_train_centernet
from train.Detrac_data_preprocess.params_init_Detrac_centernet import Eval as Detrac_config_eval_centernet

#VOC的配置文件
from train.Voc_data_preprocess.params_init_voc import TRAINING_PARAMS as Voc_config_train
from train.Voc_data_preprocess.params_init_voc import Eval as Voc_config_eval

#VOC-poly-yolo的配置文件
from train.Voc_data_preprocess.params_init_voc_poly_yolo import TRAINING_PARAMS as Voc_config_train_poly_yolo
from train.Voc_data_preprocess.params_init_voc_poly_yolo import Eval as Voc_config_eval_poly_yolo

from train.Voc_data_preprocess.params_init_voc_centernet import TRAINING_PARAMS as Voc_config_train_centernet
from train.Voc_data_preprocess.params_init_voc_centernet import Eval as Voc_config_eval_centernet

_config_factory = {
	#VOC
	"VOC":[Voc_config_train, Voc_config_eval],
	"VOC_poly_yolo":[Voc_config_train_poly_yolo, Voc_config_eval_poly_yolo],
	"VOC_centernet":[Voc_config_train_centernet, Voc_config_eval_centernet],

	#U-DETRAC
	"U-DETRAC":[Detrac_config_train, Detrac_config_eval],
	"U-DETRAC_centernet":[Detrac_config_train_centernet, Detrac_config_eval_centernet]
}

def get_config(config_char):
	return _config_factory[config_char]