# Visual Recognition using Deep Learning-Final Project
# Kaggle Competition:
https://www.kaggle.com/c/data-science-bowl-2018
## 2018 Data Science Bowl

### Find the nuclei in divergent images to advance medical discovery

#### Install
##### Environment: Detectron2
Reference: https://github.com/facebookresearch/detectron2

First, you need to install Detectron2, 

and install steps here: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

#### Preparing Datasets
After you install Detectron2, you first need to download the Dataset here:

https://github.com/samuelschen/DSB2018 (Option E: V9 dataset)

Next, because Detectron2 requires the training data set for Mask-RCNN training to be in the form of a coco dataset, we need to run Final-Project-Dataset-Preparing.py after downloading the dataset.

As long as you change the various file paths inside, you can successfully generate json files (required during training).

#### Training

Prepare the training data set, and then run Final-Project-Training.py. The various file paths and parameters in it can be changed to start training.

#### Testing

After the training is completed, weights will be stored in the output folder, and then run Final-Project-Testing.py, change the weights path, testing dataset path, etc., and you can run the final csv file.

#### Upload

Upload the csv file to https://www.kaggle.com/c/data-science-bowl-2018 and you will get a Private score.
