# YOLOV3_SUPER

#### 介绍
此仓库包含有YOLOV3－baseline以及改进版本

#### 软件架构
本程序采用python编写，采用的深度学习框架是pytorch

#### 安装需要的包：
* tqdm
* imgaug
* cython
* numpy == 1.17
* [pycocotools](git@gitee.com:yuanliangxie/cocoapi.git)

#### 安装中出现的问题

1. 安装pycocotools时需要保证已经安装cython
2. 安装pycocotools需要从这里给定的链接进行安装
```angular2
cd PythonAPI/
make
python setup.py build_ext install
```
即在本地环境中注册了pycocotools的包

#### 使用说明

本程序由几大模块构成：
* dataset模块
* model 模块
* coco_evaluater 模块用来进行构建训练时的测评器
* config模块对训练模型以及参数进行设置

如果想要在`./train/train_model`中输入额外的参数配置，则需要在里面的`trainer.set_config()`函数中引入参数


