import sys
import timm

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from os import path, makedirs, listdir

import numpy as np
import random


import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.benchmark = True
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import pandas as pd
from tqdm import tqdm
import timeit
import cv2
t0 = timeit.default_timer()

import gc



data_dir = '.'
# data_dir = '../input/hubmap-organ-segmentation'
models_folder = 'weights'
models_folder1 = 'weights'
models_folder2 = 'weights'
# models_folder = '../input/subweights0/'

df = pd.read_csv(path.join(data_dir, 'test.csv'))

organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']





def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


class TestDataset(Dataset):
    def __init__(self, df, data_dir='test_images', new_size=None):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.new_size = new_size

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img0 = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_UNCHANGED)

        orig_shape = img0.shape

        sample = {'id': r['id'], 'organ': r['organ'], 'data_source': r['data_source'], 'orig_h': orig_shape[0], 'orig_w': orig_shape[1]}

        for i in range(len(self.new_size)):

            img = cv2.resize(img0, self.new_size[i])

            img = preprocess_inputs(img)
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            sample['img{}'.format(i)] = img

        return sample


class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvSilu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)



from coat import *


class Timm_Unet(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, inp_size=3, otp_size=1, decoder_filters=[32, 48, 64, 96, 128], **kwargs):
        super(Timm_Unet, self).__init__()

        if name.startswith('coat'):
            encoder = coat_lite_medium()

            if pretrained:
                checkpoint = './weights/coat_lite_medium_384x384_f9129688.pth'
                checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                state_dict = checkpoint['model']
                encoder.load_state_dict(state_dict,strict=False)
        
            encoder_filters = encoder.embed_dims
        else:
            encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size)

            encoder_filters = [f['num_chs'] for f in encoder.feature_info]

        decoder_filters = decoder_filters

        self.conv6 = ConvSilu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvSilu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvSilu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvSilu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvSilu(decoder_filters[-3], decoder_filters[-4])

        if len(encoder_filters) == 4:
            self.conv9_2 = None
        else:
            self.conv9_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        
        self.conv10 = ConvSilu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self.cls =  nn.Linear(encoder_filters[-1] * 2, 5)
        self.pix_sz =  nn.Linear(encoder_filters[-1] * 2, 1)

        self._initialize_weights()

        self.encoder = encoder


    def forward(self, x):
        batch_size, C, H, W = x.shape

        if self.conv9_2 is None:
            enc2, enc3, enc4, enc5 = self.encoder(x)
        else:
            enc1, enc2, enc3, enc4, enc5 = self.encoder(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        if self.conv9_2 is not None:
            dec9 = self.conv9_2(torch.cat([dec9, 
                    enc1
                    ], 1))
        
        dec10 = self.conv10(dec9) # F.interpolate(dec9, scale_factor=2))

        x1 = torch.cat([F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1), 
                        F.adaptive_max_pool2d(enc5, output_size=1).view(batch_size, -1)], 1)

        # x1 = F.dropout(x1, p=0.3, training=self.training)
        organ_cls = self.cls(x1)
        pixel_size = self.pix_sz(x1)

        return self.res(dec10), organ_cls, pixel_size


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




test_batch_size = 1

# amp_autocast = suppress
amp_autocast = torch.cuda.amp.autocast

half_size = True

hubmap_only = False #True #False


organ_threshold = {
    'Hubmap': {
        'kidney'        : 90,
        'prostate'      : 100,
        'largeintestine': 80,
        'spleen'        : 100,
        'lung'          : 15,
    },
    'HPA': {
        'kidney'        : 127,
        'prostate'      : 127,
        'largeintestine': 127,
        'spleen'        : 127,
        'lung'          : 25,
    },
}


params = [
    {'size': (768, 768), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_768_e34_{}_best', models_folder, 1), 
                                    ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_768_e37_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_768_e36_{}_best', models_folder, 1), 
                                    ('coat_lite_medium', 'coat_lite_medium_768_e40_{}_best', models_folder2, 3),
                                   ],
                         'pred_dir': 'test_pred_768', 'weight': 0.2},
    {'size': (1024, 1024), 'models': [
                                      ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_1024_e32_{}_best', models_folder2, 1), 
                                      ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1024_e33_{}_best', models_folder, 1),
                                      ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1024_e38_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1024_e41_{}_best', models_folder, 3),
                                   ],
                         'pred_dir': 'test_pred_1024', 'weight': 0.3},
    {'size': (1472, 1472), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1472_e35_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1472_e39_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1472_e42_{}_best', models_folder2, 3),
                                   ],
                         'pred_dir': 'test_pred_1472', 'weight': 0.5},
]



def predict_models(param):
    print(param)

    makedirs(param['pred_dir'], exist_ok=True)

    models = []

    test_data = TestDataset(df, path.join(data_dir, 'test_images'), new_size=[param['size']])

    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=1, shuffle=False)

    torch.cuda.empty_cache()
    gc.collect()

    for model_name, checkpoint_name, checkpoint_dir, model_weight in param['models']:
        for fold in range(5):
            model = Timm_Unet(name=model_name, pretrained=None)
            snap_to_load = checkpoint_name.format(fold)
            print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(checkpoint_dir, snap_to_load), map_location='cpu')
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

            models.append((model, model_weight))


    torch.cuda.empty_cache()
    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            
            ids = sample["id"].cpu().numpy()
            orig_w = sample["orig_w"].cpu().numpy()
            orig_h = sample["orig_h"].cpu().numpy()
            # pixel_size = sample["pixel_size"].cpu().numpy()
            organ = sample["organ"]
            data_source = sample["data_source"]

            
            if hubmap_only and (data_source[0] != 'Hubmap'):
                continue


            msk_preds = []
            for i in range(0, len(ids), 1):
                msk_preds.append(np.zeros((orig_h[i], orig_w[i]), dtype='float32'))

            cnt = 0

            imgs = sample["img0"].cpu().numpy()

            with amp_autocast():
                for _tta in range(4): #8
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
                    
                    torch.cuda.empty_cache()
                    
                    for model, model_weight in models:
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
                            msk_preds[i] += model_weight * cv2.resize(msk_pred[i, 0].astype('float32'), (orig_w[i], orig_h[i]))

                    del inp
                    torch.cuda.empty_cache()


            for i in range(len(ids)):
                msk_pred = msk_preds[i] / cnt
                msk_pred = (msk_pred * 255).astype('uint8')

                print(ids[i], organ[i], res_cls[i], res_pix[i]) #pixel_size[i]

                cv2.imwrite(path.join(param['pred_dir'] , '{}.png'.format(ids[i])), msk_pred, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    del models
    torch.cuda.empty_cache()
    gc.collect()



for param in params:
    predict_models(param)



res_df = []

for _, r in df.iterrows():
    preds = []

    if hubmap_only and (r['data_source'] != 'Hubmap'):
        res_df.append({'id': r['id'], 'rle': ''})
        continue

    for param in params:
        pred = cv2.imread(path.join(param['pred_dir'], '{}.png'.format(r['id'])), cv2.IMREAD_GRAYSCALE)
        preds.append(pred * param['weight'])

    _thr = organ_threshold[r['data_source']][r['organ']]

    pred = np.asarray(preds).sum(axis=0)

    res_df.append({'id': r['id'], 'rle': rle_encode_less_memory(pred > _thr)})

    # cv2.imwrite(path.join('.', '{}.png'.format(r['id'])), pred.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 4])

res_df = pd.DataFrame(res_df)
res_df.to_csv("submission.csv", index=False)

elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))