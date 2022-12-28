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

from Dataset import TestExternalDataset


# import matplotlib.pyplot as plt
# import seaborn as sns

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

models_folder = 'weights'
models_folder1 = 'weights'

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']


data_dir = 'external_images'
out_dir = 'external_pred'


all_files = sorted(listdir(data_dir))

out_file = 'external.csv'


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(out_dir, exist_ok=True)

    cudnn.benchmark = True

    test_batch_size = 1

    # amp_autocast = suppress
    amp_autocast = torch.cuda.amp.autocast

    res_df = []

    half_size = True

    test_df = []
    fold = -1

    
    models = [[] for i in range(3)]
    
    for fold in range(5):

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

        models[0].append((model, 1))


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

        models[0].append((model, 1))


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
        
        models[0].append((model, 1))


        model = Timm_Unet(name='coat_lite_medium', pretrained=None)
        snap_to_load = 'coat_lite_medium_768_e40_{}_best'.format(fold)
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
        
        models[0].append((model, 3))




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
        
        models[1].append((model, 1))


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

        models[1].append((model, 1))


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

        models[1].append((model, 1))

        
        model = Timm_Unet(name='coat_lite_medium', pretrained=None)
        snap_to_load = 'coat_lite_medium_1024_e41_{}_best'.format(fold)
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
        
        models[1].append((model, 3))




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
        
        models[2].append((model, 2))


        model = Timm_Unet(name='tf_efficientnetv2_l_in21ft1k', pretrained=None)
        snap_to_load = 'tf_efficientnetv2_l_in21ft1k_1472_e39_{}_best'.format(fold)
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
        
        models[2].append((model, 2))


        model = Timm_Unet(name='coat_lite_medium', pretrained=None)
        snap_to_load = 'coat_lite_medium_1472_e42_{}_best'.format(fold)
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
        
        models[2].append((model, 4))

        
    test_data = TestExternalDataset(all_files, data_dir, new_size=[(768, 768), (1024, 1024), (1472, 1472)])

    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=1, shuffle=False)



    torch.cuda.empty_cache()
    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            
            ids = sample["id"]
            orig_w = sample["orig_w"].cpu().numpy()
            orig_h = sample["orig_h"].cpu().numpy()
            
            msk_preds = []
            for i in range(0, len(ids), 1):
                msk_preds.append(np.zeros((orig_h[i], orig_w[i]), dtype='float32'))

            cnt = 0

            pred_cls = []
            pred_pix = []

            for size_id in range(3):
                imgs = sample["img{}".format(size_id)].cpu().numpy()

                with amp_autocast():
                    for _tta in range(8): #8
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
                        for model, model_weight in models[size_id]:
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

                            cnt += model_weight

                            for i in range(len(ids)):
                                _ch_x = inp.shape[3] / orig_w[i]
                                _ch_y = inp.shape[2] / orig_h[i]

                                _ch_scale = _ch_x * 0.5 + _ch_y * 0.5

                                res_pix[i, 0] = res_pix[i, 0] * _ch_scale * 10

                                msk_preds[i] += model_weight * cv2.resize(msk_pred[i, 0].astype('float32'), (orig_w[i], orig_h[i]))

                            pred_pix.append(res_pix)
                            pred_cls.append(res_cls)


            pred_cls = np.asarray(pred_cls)
            pred_pix = np.asarray(pred_pix)

            pred_pix = pred_pix.mean(axis=0)
            pred_cls = pred_cls.mean(axis=0)

            for i in range(len(ids)):
                msk_pred = msk_preds[i] / cnt
                msk_pred = (msk_pred * 255).astype('uint8')

                pred_org = organs[pred_cls[i].argmax()]
                pixel_size = pred_pix[i][0]

                test_df.append({'id': ids[i], 'organ': pred_org, 'pixel_size': pixel_size, 'data_source': 'external', 'img_height': orig_h[i], 'img_width': orig_w[i], 'tissue_thickness': 0.4, 'rle': '', 'age': -1, 'sex': '',  'fold': fold})

                cv2.imwrite(path.join(out_dir, '{}.png'.format(ids[i])), msk_pred, [cv2.IMWRITE_PNG_COMPRESSION, 4])


    test_df = pd.DataFrame(test_df)
    test_df.sort_values(by='id').to_csv(out_file, index=False)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))