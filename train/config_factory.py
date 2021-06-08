#U-Detrac的配置文件
from train.Detrac_data_preprocess.params_init_Detrac import TRAINING_PARAMS as Detrac_config_train
from train.Detrac_data_preprocess.params_init_Detrac import Eval as Detrac_config_eval

from train.Detrac_data_preprocess.params_init_Detrac_centernet import TRAINING_PARAMS as Detrac_config_train_centernet
from train.Detrac_data_preprocess.params_init_Detrac_centernet import Eval as Detrac_config_eval_centernet

from train.Detrac_data_preprocess.params_init_detrac_LFFD import TRAINING_PARAMS as Detrac_config_train_lffd
from train.Detrac_data_preprocess.params_init_detrac_LFFD import Eval as Detrac_config_eval_lffd

from train.Detrac_data_preprocess.params_init_Detrac_LVnet import TRAINING_PARAMS as Detrac_config_train_LVnet
from train.Detrac_data_preprocess.params_init_Detrac_LVnet import Eval as Detrac_config_eval_LVnet

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_deconv import TRAINING_PARAMS as Detrac_config_train_LVnet_deconv
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_deconv import Eval as Detrac_config_eval_LVnet_deconv

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_deconv_centerloss import TRAINING_PARAMS as Detrac_config_train_LVnet_deconv_centerloss
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_deconv_centerloss import Eval as Detrac_config_eval_LVnet_deconv_centerloss

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_pure_centerloss import TRAINING_PARAMS as Detrac_config_train_LVnet_pure_centerloss
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_pure_centerloss import Eval as Detrac_config_eval_LVnet_pure_centerloss

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_iou_assign import TRAINING_PARAMS as Detrac_config_train_LVnet_iou_assign
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_iou_assign import Eval as Detrac_config_eval_LVnet_iou_assign

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_iou_assign_no_fpn import TRAINING_PARAMS as Detrac_config_train_LVnet_iou_assign_no_fpn
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_with_iou_assign_no_fpn import Eval as Detrac_config_eval_LVnet_iou_assign_no_fpn

from train.Detrac_data_preprocess.params_init_Detrac_tiny_yolov3 import TRAINING_PARAMS as Detrac_config_train_tiny_yolov3
from train.Detrac_data_preprocess.params_init_Detrac_tiny_yolov3 import Eval as Detrac_config_eval_tiny_yolov3

from train.Detrac_data_preprocess.params_init_Detrac_mobilev2 import TRAINING_PARAMS as Detrac_config_train_mobilev2_yolov3
from train.Detrac_data_preprocess.params_init_Detrac_mobilev2 import Eval as Detrac_config_eval_mobilev2_yolov3

from train.Detrac_data_preprocess.params_init_Detrac_shulffnetv2_yolov3 import TRAINING_PARAMS as Detrac_config_train_shulffnetv2_yolov3
from train.Detrac_data_preprocess.params_init_Detrac_shulffnetv2_yolov3 import Eval as Detrac_config_eval_shulffnetv2_yolov3

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_fpn_large_weight import TRAINING_PARAMS as Detrac_config_train_LVnet_fpn_large_weight
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_fpn_large_weight import Eval as Detrac_config_eval_LVnet_fpn_large_weight

from train.Detrac_data_preprocess.params_init_Detrac_LVnet_fpn_largest_weight import TRAINING_PARAMS as Detrac_config_train_LVnet_fpn_largest_weight
from train.Detrac_data_preprocess.params_init_Detrac_LVnet_fpn_largest_weight import Eval as Detrac_config_eval_LVnet_fpn_largest_weight

#VOC的配置文件
from train.Voc_data_preprocess.params_init_voc import TRAINING_PARAMS as Voc_config_train
from train.Voc_data_preprocess.params_init_voc import Eval as Voc_config_eval

#VOC-poly-yolo的配置文件
from train.Voc_data_preprocess.params_init_voc_poly_yolo import TRAINING_PARAMS as Voc_config_train_poly_yolo
from train.Voc_data_preprocess.params_init_voc_poly_yolo import Eval as Voc_config_eval_poly_yolo

from train.Voc_data_preprocess.params_init_voc_centernet import TRAINING_PARAMS as Voc_config_train_centernet
from train.Voc_data_preprocess.params_init_voc_centernet import Eval as Voc_config_eval_centernet

#ATR-SKY
from train.ATR_data_preprocess.params_init_ATR import TRAINING_PARAMS as ATR_config_train_yolov3
from train.ATR_data_preprocess.params_init_ATR import Eval as ATR_config_eval_yolov3

from train.ATR_data_preprocess.params_init_ATR_yolov5 import TRAINING_PARAMS as ATR_config_train_yolov5
from train.ATR_data_preprocess.params_init_ATR_yolov5 import Eval as ATR_config_eval_yolov5

from train.ATR_data_preprocess.params_init_ATR_centernet_resnet18 import TRAINING_PARAMS as ATR_config_train_Centernet_resnet18
from train.ATR_data_preprocess.params_init_ATR_centernet_resnet18 import Eval as ATR_config_eval_Centernet_resnet18

from train.ATR_data_preprocess.params_init_ATR_centernet_hourglass import TRAINING_PARAMS as ATR_config_train_Centernet_hourglass
from train.ATR_data_preprocess.params_init_ATR_centernet_hourglass import Eval as ATR_config_eval_Centernet_hourglass

_config_factory = {
	#VOC
	"VOC":[Voc_config_train, Voc_config_eval],
	"VOC_poly_yolo":[Voc_config_train_poly_yolo, Voc_config_eval_poly_yolo],
	"VOC_centernet":[Voc_config_train_centernet, Voc_config_eval_centernet],

	#U-DETRAC
	"U-DETRAC":[Detrac_config_train, Detrac_config_eval],
	"U-DETRAC_centernet":[Detrac_config_train_centernet, Detrac_config_eval_centernet],
	"U-DETRAC_lffd":[Detrac_config_train_lffd, Detrac_config_eval_lffd],
	"U-DETRAC_LVnet":[Detrac_config_train_LVnet, Detrac_config_eval_LVnet],
	"U-DETRAC_LVnet_deconv":[Detrac_config_train_LVnet_deconv, Detrac_config_eval_LVnet_deconv],
	"U-DETRAC_LVnet_deconv_centerloss":[Detrac_config_train_LVnet_deconv_centerloss, Detrac_config_eval_LVnet_deconv_centerloss],
	"U-DETRAC_LVnet_pure_centerloss":[Detrac_config_train_LVnet_pure_centerloss, Detrac_config_eval_LVnet_pure_centerloss],
	"U-DETRAC_LVnet_iou_assign":[Detrac_config_train_LVnet_iou_assign, Detrac_config_eval_LVnet_iou_assign],
	"U-DETRAC_LVnet_iou_assign_no_fpn":[Detrac_config_train_LVnet_iou_assign_no_fpn, Detrac_config_eval_LVnet_iou_assign_no_fpn],
	"U-DETRAC_tiny_yolov3":[Detrac_config_train_tiny_yolov3, Detrac_config_eval_tiny_yolov3],
	"U-DETRAC_mobilev2_yolov3": [Detrac_config_train_mobilev2_yolov3, Detrac_config_eval_mobilev2_yolov3],
	"U-DETRAC_shulffnetv2_yolov3":[Detrac_config_train_shulffnetv2_yolov3, Detrac_config_eval_shulffnetv2_yolov3],
	"U-DETRAC_LVnet_fpn_large_weight":[Detrac_config_train_LVnet_fpn_large_weight, Detrac_config_eval_LVnet_fpn_large_weight],
	"U-DETRAC_LVnet_fpn_largest_weight":[Detrac_config_train_LVnet_fpn_largest_weight, Detrac_config_eval_LVnet_fpn_largest_weight],

	#ATR-SKY
	"ATR-SKY_YOLOV3":[ATR_config_train_yolov3, ATR_config_eval_yolov3],
	"ATR-SKY_YOLOV5":[ATR_config_train_yolov5, ATR_config_eval_yolov5],
	"ATR-SKY_Centernet18":[ATR_config_train_Centernet_resnet18, ATR_config_eval_Centernet_resnet18],
	"ATR-SKY_Centernet_hourglass":[ATR_config_train_Centernet_hourglass, ATR_config_eval_Centernet_hourglass],
}

def get_config(config_char):
	return _config_factory[config_char]