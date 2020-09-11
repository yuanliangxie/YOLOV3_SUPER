from dataset.detrac_dataset_add_SSD_aug import DetracDataset
from dataset.voc_dataset_add_SSD_aug import VOCDataset


_dataset_factory = {
	"VOC": VOCDataset,
	"U-DETRAC": DetracDataset,
}


def load_dataset(config_train, image_size_train):
	if config_train["config_name"][0:3] == "VOC":
		Dataset = _dataset_factory["VOC"]
		dataset = Dataset(list_path= config_train["train_path"], labels_path= config_train["train_labels_path"],
						  img_size= image_size_train, is_training=True, batch_size=config_train["batch_size"])

	elif config_train["config_name"] == "U-DETRAC":
		Dataset = _dataset_factory["U-DETRAC"]
		dataset = Dataset(list_path=config_train["train_path"], ignore_region_path=config_train["train_ignore_region"],
		                        labels_path=config_train["train_labels_path"],
		                        img_size=image_size_train, is_training=True, batch_size=config_train["batch_size"])

	else:
		return None
	return dataset
