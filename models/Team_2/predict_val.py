import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from models import Timm_Unet

from Dataset import TestDataset

# import matplotlib.pyplot as plt
# import seaborn as sns


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

models_folder = 'weights'
models_folder1 = 'weights'

df = pd.read_csv('folds.csv')

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

masks_dir = 'masks'
data_dir = 'train_images'
test_dir = 'test_images'


out_dir = 'train_pred_oof'


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(out_dir, exist_ok=True)

    cudnn.benchmark = True

    test_batch_size = 1

    amp_autocast = torch.cuda.amp.autocast

    for fold in range(5): 
        print('predicting fold', fold)

        test_df = df[df['fold'] == fold].copy()
        test_df.reset_index(drop=True, inplace=True)
        
        test_df = test_df
        test_df.reset_index(drop=True, inplace=True)

        test_data = TestDataset(test_df, data_dir, new_size=[(768, 768), (1024, 1024), (1472, 1472)])

        test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=2, shuffle=False)

        half_size = True
        
        models = [[] for i in range(3)]
            

        model = Timm_Unet(name='tf_efficientnet_b7_ns', pretrained=None)
        snap_to_load = 'tf_efficientnet_b7_ns_768_e34_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder1, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()

        models[0].append(model)


        model = Timm_Unet(name='convnext_large_384_in22ft1k', pretrained=None)
        snap_to_load = 'convnext_large_384_in22ft1k_768_e37_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder1, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()

        models[0].append(model)


        model = Timm_Unet(name='tf_efficientnetv2_l_in21ft1k', pretrained=None)
        snap_to_load = 'tf_efficientnetv2_l_in21ft1k_768_e36_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()
        
        models[0].append(model)




        model = Timm_Unet(name='tf_efficientnetv2_l_in21ft1k', pretrained=None)
        snap_to_load = 'tf_efficientnetv2_l_in21ft1k_1024_e38_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()
        
        models[1].append(model)


        model = Timm_Unet(name='convnext_large_384_in22ft1k', pretrained=None)
        snap_to_load = 'convnext_large_384_in22ft1k_1024_e32_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder1, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()

        models[1].append(model)


        model = Timm_Unet(name='tf_efficientnet_b7_ns', pretrained=None)
        snap_to_load = 'tf_efficientnet_b7_ns_1024_e33_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder1, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()

        models[1].append(model)




        model = Timm_Unet(name='tf_efficientnet_b7_ns', pretrained=None)
        snap_to_load = 'tf_efficientnet_b7_ns_1472_e35_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()
        
        models[2].append(model)


        model = Timm_Unet(name='tf_efficientnetv2_l_in21ft1k', pretrained=None)
        snap_to_load = 'tf_efficientnetv2_l_in21ft1k_1472_e29_{}_best'.format(fold)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
            checkpoint['epoch'], checkpoint['best_score']))
        model = model.eval().cuda()
        
        models[2].append(model)



        torch.cuda.empty_cache()
        with torch.no_grad():
            for sample in tqdm(test_data_loader):
                
                ids = sample["id"].cpu().numpy()
                orig_w = sample["orig_w"].cpu().numpy()
                orig_h = sample["orig_h"].cpu().numpy()
                # pixel_size = sample["pixel_size"].cpu().numpy()
                organ = sample["organ"]
                # data_source = sample["data_source"]


                msk_preds = []
                for i in range(0, len(ids), 1):
                    msk_preds.append(np.zeros((orig_h[i], orig_w[i]), dtype='float32'))

                cnt = 0

                for size_id in range(3):
                    imgs = sample["img{}".format(size_id)].cpu().numpy()

                    with amp_autocast():
                        for _tta in range(8):
                            _i = _tta // 2
                            _flip = False
                            if _tta % 2 == 1:
                                _flip = True

                            if _i == 0:
                                inp = imgs.copy()
                            elif _i == 1:
                                inp = np.rot90(imgs, k=1, axes=(2,3)).copy()
                            elif _i == 2:
                                inp = np.rot90(imgs, k=2, axes=(2,3)).copy()
                            elif _i == 3:
                                inp = np.rot90(imgs, k=3, axes=(2,3)).copy()

                            if _flip:
                                inp = inp[:, :, :, ::-1].copy()

                            inp = torch.from_numpy(inp).float().cuda()                   
                            
                            # torch.cuda.empty_cache()
                            for model in models[size_id]:
                                out, res_cls, res_pix = model(inp)
                                msk_pred = torch.sigmoid(out).cpu().numpy()
                                
                                res_cls = torch.softmax(res_cls, dim=1).cpu().numpy()
                                res_pix = res_pix.cpu().numpy()
                                
                                if _flip:
                                    msk_pred = msk_pred[:, :, :, ::-1].copy()

                                if _i == 1:
                                    msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                                elif _i == 2:
                                    msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                                elif _i == 3:
                                    msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                                cnt += 1

                                for i in range(len(ids)):
                                    msk_preds[i] += cv2.resize(msk_pred[i, 0].astype('float32'), (orig_w[i], orig_h[i]))

                                # torch.cuda.empty_cache()

                for i in range(len(ids)):

                    msk_pred = msk_preds[i] / cnt
                    msk_pred = (msk_pred * 255).astype('uint8')

                    msk_pred = msk_pred.astype('uint8')

                    cv2.imwrite(path.join(out_dir, '{}.png'.format(ids[i])), msk_pred, [cv2.IMWRITE_PNG_COMPRESSION, 4])


    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))