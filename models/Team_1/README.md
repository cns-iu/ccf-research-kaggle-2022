Below you can find an outline of how to reproduce team 1 solution for the HuBMAP + HPA - Hacking the Human Body competition.  

# CONTENTS  
notebooks_for_reference: for reference of cell outputs.  
multi_class_dataset: datasets for training  
models: Trained 6 models  
model_configs: MMSegmentation model configuration.  
readme.md
SETTINGS.json  
segformer_training.ipynb: For reproducing training models on Colab.  
predict.py  
inference_sample_on_colab.ipynb  
A kaggle notebook "verification_hubmap" for inference on Kaggle.  
A kaggle notebook "mmseg_src" for using required libraries offline on kaggle notebooks. 
  
# Hardware requirements:  
The models used Google Colab Pro+ for training and Kaggle Kernels for inference.
  
# MODEL BUILD:  
1) trained model prediction  
    a) expect this to run in 1.5h  
    b) uses binary model files  
    c) command: python ./predict.py in the shared kaggle notebook "verification_hubamp".  
2) retrain models  
    a) expect this to run about 4.5h per model (It depends on GPU. It will cost over 6h in case of Tesla P100)  
    b) trains all models from scratch  
    c) follow this with (2) to produce entire solution from scratch  
    d) command: kaggle notebook execution  
  
# Model Reference  
  
segformer_mit-b3_1024  
model: mit-b3 (used preptrained encorder:mit-b3)  
image size: 1024x1024  
seed:0  
  
segformer_mit-b4_960  
This is the best single model.   
model: mit-b4 (used preptrained encorder:mit-b4)  
image size: 960x960  
seed:0  

segformer_mit-b4_960_2  
model: mit-b4 (used pretrained segformer_mit-b4 trained on cityscape 1024x1024)  
image size: 960x960  
seed:0  
  
segformer_mit-b5_928  
model: mit-b5 (used pretrained encorder:mit-b5)  
image size: 928x928  
seed:0  

segformer_mit-b5_960  
model: mit-b5 (used pretrained segformer_mit-b5 trained on cityscape 1024x1024)  
image size: 960x960  
seed:0  
GPU: NVIDIA Tesla P100 in this notebook due to out of memory in Tesla V100.   
  
segformer_mit-b5_960_2  
model: mit-b5 (used pretrained segformer_mit-b5 trained on cityscape 1024x1024)  
dataset: Additional dataset "images_stained_with_pas" and "images_stained_with_pas2"  
image size: 960x960  
seed:2022  
GPU: NVIDIA Tesla P100 in this notebook due to out of memory in Tesla V100.  
