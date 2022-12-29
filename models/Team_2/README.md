Below you can find a outline of how to reproduce my solution for the "HuBMAP + HPA - Hacking the Human Body" competition.

Note: competition data expected to be extracted in current directory. All data shoould be placed in current directory.


# ARCHIVE CONTENTS:
weights: models weights.  
train.sh: executes training of all models.  
train.py: code to train single model.  
predict.py: generates submission file for provided test.csv (all data in current directory). The same as submission notebook.  
Dataset.py: Dataset class, augmentations here.  
create_masks.py: create masks for train.  
create_folds.py: split to folds and get folds.csv.  
predict_external.py: predict external images and get pseudo labels (external_images and external_pred folders).  
color_transfer.py: transfer colors for train_images, three versions saved to train_color_transfered (~14GB). Will be executed with train.sh.  
models.py: used model code. Timm_Unet.  
predict_val.py: predict oof for train_images to train_pred_oof (for pseudo).  
hpa_download.py: download additional HPA images hpa_images_extra.  


# HARDWARE: (The following specs were used to create the original solution)
System with 2 x NVIDIA RTX A6000 (48 GB). Large GPU memory required for high resolution models.


# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Ubuntu 22.04 LTS
Python 3.9 (Anaconda installation)
CUDA 11.6
torch and other libs from requirements.txt


# DATA SETUP
Competition data expected to be extracted in current directory. All data shoould be placed in current directory.

hpa_images_extra, hpa_images_extra_pred: extra images from HPA and predictions from first versions of ensemble.  
external_images, external_pred: hand picked extra images from sources in forum (prev competitions, cancerimagingarchive.net) and predictions from first versions of ensemble.  
masks: generated masks for train.  
train_pred_oof: oof prediction for train, with 30% chance take it instead of mask during training.  
train_color_transfered: result of color_transfer.py ~14GB.  


1) ordinary prediction (overwrites submission.csv)  
`python predict.py`

2) retrain all models (overwrites models in weights directory)  
`sh ./train.sh`
