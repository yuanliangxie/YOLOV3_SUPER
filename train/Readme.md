####此yolov3复现程序由yuanliangxie编写
运行此项目按如下步骤：
#####1.在./train/params_init_voc.py中填写训练参数
#####2.运行voc_data_process.py运行程序生成训练图片的链接和标签
#####3.在./dataset/voc_dataset_add_SSD_aug.py中运行程序对训练数据进行可视化
#####4.在./select_device/params_device中选择cpu或gpu
#####5.运行./train/train.py进行训练
#####6.测评程序在./evaluate/eval_voc.py,需要配置./evaluate/yolov3_config_voc_test.py文件
#####7.demo程序在./test_video/yolov3_test_interface.py, 需要配置./test_video/params_interface_voc.py文件