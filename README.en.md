# YOLOV3-SUPER
---
# Introduction
This is a YOLOV3 baseline written in PyTorch. The eval dataset used is PASCAL VOC(not use difficulty). The eval tool is the [cocoAPI](https://gitee.com/yuanliangxie/cocoapi). The mAP gains the score as same as the original paper.
Subsequently, we will continue to update the code by adding some new and efficient methods to make it more concise, lighter, faster in vehicle detection.
## Results


| name | Train Dataset | Val Dataset | mAP(mine) | notes |
| :----- | :----- | :------ | :----- | :-----|
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.835 | \+cosine lr + multi-scale + smooth_L1 |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | cocoAPI | 0.821 | \+cosine lr + multi-scale + smooth_L1|
 
## FPS
| Implement | Backbone | Input Size | Batch Size |Inference Time | FPS |
| :----- | :----- | :------ | :----- | :-----| :-----|
| Paper | Darknet53 | 320 | 1 | 22ms |  45 |
| Paper | Darknet53 | 416 | 1 | 29ms |  34 |
| Paper | Darknet53 | 608 | 1 | 51ms |  19 |
| Our | Darknet53 | 320 | 1 | 23ms |  43 |
| Our | Darknet53 | 416 | 1 | 27ms |  36 |
| Our | Darknet53 | 608 | 1 | 40ms |  29 |

`Note` : 

* YOLOV3-\*-544 means test image size is 544. `"*"` means the multi-scale.
* In the test, the nms threshold is 0.5 and the conf_score is 0.01.
* Now only support the single gpu to train and test.

---
## Environment
`Note`: This is just a reference, the environment doesn’t have to be exactly the same.
* Nvidia GeForce RTX 1080 Ti
* CUDA10.1
* CUDNN7.6.0
* ubuntu 18.04

```bash
# install packages
pip install -r requirements.txt 
```

---
## Brief

* [x] Data Augmentation (SSD Augmentation methods)
* [x] Step lr Schedule 
* [x] Multi-scale Training (320 to 608)
* [x] cosine lr


---
## Prepared work

### 1、Git clone YOLOV3 repository
```Bash
git clone https://gitee.com/vehicle_1/YOLOV3_SUPER.git
```

### 2、Download dataset
#### 2.1 PASCAL_VOC
* Download Pascal VOC dataset : [VOC data.zip](https://pan.baidu.com/s/1PXnqvQCQn5IWRRei1ImDkQ)  Passwords: 7vfh
* The organizational structure of this data will be presented below:<br>
|--voc_data<br>
|--|--VOCtest-2007<br>
|--|--|--VOCdevkit<br>
|--|--|--|--VOC2007<br>
|--|--VOCtrainval-2007<br>
|--|--|--VOCdevkit<br>
|--|--|--|--VOC2007<br>
|--|--VOCtrainval-2012<br>
|--|--|--VOCdevkit<br>
|--|--|--|--VOC2012<br>

#### 2.2 U-DETRAC
* Download Pascal VOC dataset : [U-DETRAC]()
* The organizational structure of this data will be presented below:<br>
|--U-DETRA_dataset<br>
|--|--DETRAC-Test-Annotations-XML<br>
|--|--DETRAC-Train-Annotations-XML<br>
|--|--DETRAC-test-data<br>
|--|--DETRAC-train-data<br>

#### Prepare to do
* first<br>
we should first adjust the config params in `./train/Detrac_data_preprocess/params_init_Detrac` or `./train/Voc_data_preprocess/params_init_voc`<br>
We should pay attention to the settings of these parameters:<br>
*"data_path"
*"working_dir"
*"DATA_PATH"
*"PROJECT_PATH"<br>
* second<br>
```Bash
cd train/Detrac_data_preprocess
python Detrac_data_process.py
```
or
```Bash
cd train/Voc_data_preprocess
python voc_data_process.py
```
then the dataset is ready
* third<br>
```Bash
cd evaluate/evaluate_coco
python voc_convert2_coco.py --xml_dir ~~~
```
or
```Bash
cd evaluate/evaluate_detrac_coco_api
python Detrac_convert2_coco.py
```
then the evaluation's gt_json for evaluate is ready


### 3、Download weight file
* Darknet pre-trained weight :  [darknet53.conv.74]() 
* Darknet COCO-pretrained pth : [darknet53_weights_pytorch.pth]()

Make dir `weight/` in the YOLOV3 and put the weight file in.

---
## Train

Run the following command to start training and see the details in the `darknet53/name/time_id/log.txt`

```Bash

cd train/
python train_model.py --config_name "VOC" --device_id 0 --config_model_name "yolov3"

```

---
## Test
You should config `evaluate/evaluate_coco/yolov3_config_voc_test.py` or `evaluate/evaluate_detrac_coco_api/yolov3_config_dtrac_test.py`
```Bash
cd evaluate/evaluate_coco/
python coco_eval_voc.py
```

```Bash
cd evaluate/evaluate_detrac_coco_api/
python  coco_eval_detrac.py
```
The images can be seen in the `evaluate/evaluate_coco/data/results`or`evaluate/evaluate_detrac_coco_api/data/results`

---
## TODO

* [ ] model compression
* [ ] focal loss
* [ ] GIOU
* [ ] Label smooth
* [ ] Mixup
* [ ] Mobilenet v1-v3
* [ ] Mossaic
* [ ] Poly-yolo
* [ ] Gaussian-yolo
* [ ] centernet-Gaussian kernel

---
## Reference

* pytorch : https://github.com/ultralytics/yolov3
* pytorch : https://github.com/Peterisfar/YOLOV3

