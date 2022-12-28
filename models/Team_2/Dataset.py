from os import path

import torch
from torch.utils.data import Dataset

import numpy as np
import random

import cv2

import pandas as pd

from utils import *

from imgaug import augmenters as iaa

# import matplotlib.pyplot as plt
# import seaborn as sns


organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

hubmap_pix_sizes = {
            'kidney': 0.5,
            'prostate': 6.263,
            'largeintestine': 0.229,
            'spleen': 0.4945,
            'lung': 0.7562
        }

        
organ_threshold = {
    'external': { #Hubmap
        'kidney'        : 90,
        'prostate'      : 100,
        'largeintestine': 80,
        'spleen'        : 100,
        'lung'          : 15,
    },
    'HPA': {
        'kidney'        : 100,
        'prostate'      : 100,
        'largeintestine': 100,
        'spleen'        : 100,
        'lung'          : 25,
    },
}



class TrainDataset(Dataset):
    def __init__(self, df, data_dir='train_images', masks_dir='masks',  aug=True, new_size=None, epoch_size=-1, mask_half=True):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.aug = aug
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.masks_dir = masks_dir
        self.new_size = new_size
        self.epoch_size = len(self.df)
        self.mask_half = mask_half
        if epoch_size > 0:
            self.epoch_size = epoch_size

        self.ext_masks_dir = 'external_pred'
        self.ext_dir = 'external_images'

        self.masks_pred_dir = 'train_pred_oof'

        self.train_colored = 'train_color_transfered'

        self.df_organs = []
        for o in organs:
            self.df_organs.append(df[df['organ'] == o].reset_index())
            
        self.df_hpa_extra = pd.read_csv('hpa_images_extra.csv')

        self.df_organs_hpa_extra = []
        for o in organs:
            self.df_organs_hpa_extra.append(self.df_hpa_extra[self.df_hpa_extra['organ'] == o].reset_index())


    def __len__(self):
        return self.epoch_size


    def __getitem__(self, idx):
        try_again = True
        img = None
        msk = None
        r = None

        while try_again:
            try_again = False

            try:
                _o_idx = idx % len(organs)
                
                _idx = random.randrange(len(self.df_organs[_o_idx]))
                r = self.df_organs[_o_idx].iloc[_idx]

                pixel_size = r['pixel_size']

                if (r['data_source'] != 'external') and (random.random() < 0.2):
                    
                    _idx = random.randrange(len(self.df_organs_hpa_extra[_o_idx]))

                    r = self.df_organs_hpa_extra[_o_idx].iloc[_idx]

                    pixel_size = r['pixel_size']

                    if random.random() > 0.6:
                        pixel_size = 0.4

                    img = cv2.imread(path.join('hpa_images_extra', '{}'.format(r['id'])), cv2.IMREAD_COLOR)
                    msk = cv2.imread(path.join('hpa_images_extra_pred', '{}.png'.format(r['id'])), cv2.IMREAD_GRAYSCALE)

                    _thr = organ_threshold['HPA'][r['organ']]
                    if random.random() > 0.6:
                        _thr = 127
                    msk = ((msk > _thr) * 255).astype('uint8')
                    
                else:
                    if r['data_source'] == 'external':
                        img = cv2.imread(path.join(self.ext_dir, '{}'.format(r['id'])), cv2.IMREAD_COLOR)
                        msk = cv2.imread(path.join(self.ext_masks_dir, '{}.png'.format(r['id'])), cv2.IMREAD_GRAYSCALE)

                        _thr = organ_threshold['external'][r['organ']]
                        if random.random() > 0.7:
                            _thr = 127
                        msk = ((msk > _thr) * 255).astype('uint8')
                    else:
                        if random.random() > 0.15:
                            img = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_UNCHANGED)
                        else:
                            k = random.randrange(3)
                            img = cv2.imread(path.join(self.train_colored, '{}_{}.png'.format(r['id'], k)), cv2.IMREAD_UNCHANGED)
                            

                        if random.random() > 0.3:
                            msk = cv2.imread(path.join(self.masks_dir, '{}.png'.format(r['id'])), cv2.IMREAD_UNCHANGED)
                        else:
                            msk = cv2.imread(path.join(self.masks_pred_dir, '{}.png'.format(r['id'])), cv2.IMREAD_UNCHANGED)

                            _thr = organ_threshold['HPA'][r['organ']]
                            if random.random() > 0.6:
                                _thr = 127
                            msk = ((msk > _thr) * 255).astype('uint8')

            except Exception as ex:
                try_again = True
                print('Exception occured: {}. File: {}'.format(ex, r['id']))
                
            if img is None:
                try_again = True
                print('img is None. File: {}'.format(r['id']))

            if msk is None:
                try_again = True
                print('msk is None. File: {}'.format(r['id']))

            if try_again:
                idx = random.randint(0, self.epoch_size - 1)
                continue

            
            if self.aug:
                _p = 0.5
                if (r['data_source'] == 'HPA') and  (random.random() > _p):
                    _ch_scale = 0.4 / hubmap_pix_sizes[r['organ']]
                    pixel_size = hubmap_pix_sizes[r['organ']]
                    
                    _h = int(img.shape[0] * _ch_scale)
                    _w = int(img.shape[1] * _ch_scale)

                    img = cv2.resize(img, (_w, _h), interpolation=cv2.INTER_LANCZOS4)
                    msk = cv2.resize(msk, (_w, _h), self.new_size)
                    

                _p = 0.1
                if random.random() > _p:
                    sz = random.randrange(int(img.shape[0] * 0.25))
                    if sz > 0:
                        if random.random() > 0.5:
                            img = img[sz:-sz, :, :]
                            msk = msk[sz:-sz, :]
                        else:
                            _v = random.choice([0, 255])
                            img = np.pad(img, ((sz, sz), (0, 0), (0, 0)), constant_values=_v)
                            msk = np.pad(msk, ((sz, sz), (0, 0)), constant_values=0)

                if random.random() > _p:
                    sz = random.randrange(int(img.shape[0] * 0.25))
                    if sz > 0:
                        if random.random() > 0.5:
                            img = img[:, sz:-sz, :]
                            msk = msk[:, sz:-sz]
                        else:
                            _v = random.choice([0, 255])
                            img = np.pad(img, ((0, 0), (sz, sz), (0, 0)), constant_values=_v)
                            msk = np.pad(msk, ((0, 0), (sz, sz)), constant_values=0)


            if self.new_size is not None:
                _ch_x = img.shape[1] / self.new_size[0]
                _ch_y = img.shape[0] / self.new_size[1]
                _ch_scale = _ch_x * 0.5 + _ch_y * 0.5

                pixel_size = pixel_size * _ch_scale

                img = cv2.resize(img, self.new_size)
                msk = cv2.resize(msk, self.new_size)


            if self.aug:
                _p = 0.5
                if random.random() > _p:
                    img = img[:, ::-1, :]
                    msk = msk[:, ::-1]
                
                _p = 0
                if random.random() > _p:
                    _k = random.randrange(4)
                    img = np.rot90(img, k=_k, axes=(0,1))
                    msk = np.rot90(msk, k=_k, axes=(0,1))


                _p = 0.01
                if random.random() > _p:
                    shift_pnt = (random.randint(-64, 64), random.randint(-64, 64))
                    _v = random.choice([0, 255])
                    img = shift_image(img, shift_pnt, borderValue=(_v, _v, _v))
                    msk = shift_image(msk, shift_pnt, borderValue=0)
            
                _p = 0.01
                if random.random() > _p:
                    _d = int(img.shape[0] * 0.2)
                    rot_pnt =  (img.shape[0] // 2 + random.randint(-_d, _d), img.shape[1] // 2 + random.randint(-_d, _d))
                    scale = 1
                    if random.random() > 0.01:
                        scale = random.normalvariate(1.0, 0.35)

                    pixel_size = pixel_size * scale

                    angle = 0
                    if random.random() > 0.01:
                        angle = random.randint(0, 90) - 45
                    if (angle != 0) or (scale != 1):
                        _v = random.choice([0, 255])
                        img = rotate_image(img, angle, scale, rot_pnt, borderValue=(_v, _v, _v))
                        msk = rotate_image(msk, angle, scale, rot_pnt, borderValue=0)


                _p = 0.5
                if random.random() > _p:
                    img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

                _p = 0.5
                if random.random() > _p:
                    img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

                _p = 0.92
                if random.random() > _p:
                    img = img[:, :, ::-1]

                _p = 0.5
                if random.random() > _p:
                    if random.random() > 0.66:
                        img = clahe(img)
                    elif random.random() > 0.5:
                        img = gauss_noise(img)
                    elif random.random() > 0:
                        img = cv2.blur(img, (3, 3))
                
                _p = 0.5
                if random.random() > _p:
                    if random.random() > 0.66:
                        img = saturation(img, 0.8 + random.random() * 0.4)
                    elif random.random() > 0.5:
                        img = brightness(img, 0.8 + random.random() * 0.4)
                    elif random.random() > 0:
                        img = contrast(img, 0.8 + random.random() * 0.4)
                
                _p = 0.9 #0.9
                if random.random() > _p:
                    el_det = self.elastic.to_deterministic()
                    img = el_det.augment_image(img)

                _p = 0.8 # 0.92
                if random.random() > _p:
                    sz = random.randrange(int(img.shape[0] * 0.2))
                    if sz > 0:
                        _v = random.choice([0, 255])
                        img[:sz, :, :] = _v
                        msk[:sz, :,] = 0
                        img[-sz:, :, :] = _v
                        msk[-sz:, :] = 0
                if random.random() > _p:
                    sz = random.randrange(int(img.shape[0] * 0.2))
                    if sz > 0:
                        _v = random.choice([0, 255])
                        img[:, :sz, :] = _v
                        msk[:, :sz] = 0
                        img[:, -sz:, :] = _v
                        msk[:, -sz:] = 0

                _p = 0.8
                if random.random() > _p:
                    for _i in range(random.randrange(5)):
                        _v = random.choice([0, 255])
                        sz0 = random.randrange(1, int(img.shape[0] * 0.3))
                        sz1 = random.randrange(1, int(img.shape[1] * 0.3))
                        x0 = random.randrange(img.shape[1] - sz1)
                        y0 = random.randrange(img.shape[0] - sz0)
                        img[y0:y0+sz0, x0:x0+sz1, :] = _v
                        msk[y0:y0+sz0, x0:x0+sz1] = 0



            if (msk > 127).sum() == 0: 
                try_again = True
                idx = random.randint(0, self.epoch_size - 1)


        if self.mask_half:
            msk = cv2.resize(msk, (msk.shape[1] // 2, msk.shape[0] // 2))


        msk = (msk > 127)
        msk = msk[:, :, np.newaxis]

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        pixel_size /= 10.0
        pixel_size = torch.from_numpy(np.array([pixel_size])).float()

        lbl_bin = np.zeros((5,), dtype=int)
        lbl_bin[_o_idx] = 1
        lbl_bin = torch.from_numpy(lbl_bin).float()

        sample = {'img': img, 'msk': msk, 'id': str(r['id']), 'organ': r['organ'], 'pixel_size': pixel_size, 'lbl': _o_idx, 'lbl_bin': lbl_bin}

        return sample



class ValDataset(Dataset):
    def __init__(self, df, data_dir='train_images', masks_dir='masks', new_size=None, mask_half=True):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.new_size = new_size
        self.epoch_size = len(self.df) * 4
        self.mask_half = mask_half

        self.ext_masks_dir = 'external_pred'
        self.ext_dir = 'external_images'


    def __len__(self):
        return self.epoch_size


    def __getitem__(self, idx):
        r = self.df.iloc[idx % len(self.df)]

        tta = idx // len(self.df)

        if r['data_source'] == 'external':
            img = cv2.imread(path.join(self.ext_dir, '{}'.format(r['id'])), cv2.IMREAD_COLOR)
            msk = cv2.imread(path.join(self.ext_masks_dir, '{}.png'.format(r['id'])), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(path.join(self.masks_dir, '{}.png'.format(r['id'])), cv2.IMREAD_UNCHANGED)

        _thr = organ_threshold[r['data_source']][r['organ']]
        msk = ((msk > _thr) * 255).astype('uint8')

        orig_shape = img.shape

        pixel_size = r['pixel_size']

        if tta > 0:
            _ch_scale = 0.4 / hubmap_pix_sizes[r['organ']]
            pixel_size = hubmap_pix_sizes[r['organ']]
            
            _h = int(img.shape[0] * _ch_scale)
            _w = int(img.shape[1] * _ch_scale)

            img = cv2.resize(img, (_w, _h), interpolation=cv2.INTER_LANCZOS4)
            msk = cv2.resize(msk, (_w, _h), self.new_size)

        #tta
        if tta == 1:
            img = img[:, ::-1, :]
            msk = msk[:, ::-1]

        if tta == 2:
            img = img[::-1, ::-1, :]
            msk = msk[::-1, ::-1]

            sz = int(img.shape[0] * 0.1)
            img = img[sz:-sz, sz:-sz, :]
            msk = msk[sz:-sz, sz:-sz]

        if tta == 3:
            img = img[::-1, :, :]
            msk = msk[::-1, :]

            sz = int(img.shape[0] * 0.1)
            img = np.pad(img, ((sz, sz), (sz, sz), (0, 0)), constant_values=255)
            msk = np.pad(msk, ((sz, sz), (sz, sz)), constant_values=0)


        if self.new_size is not None:
            _ch_x = img.shape[1] / self.new_size[0]
            _ch_y = img.shape[0] / self.new_size[1]
            _ch_scale = _ch_x * 0.5 + _ch_y * 0.5

            pixel_size = pixel_size * _ch_scale

            img = cv2.resize(img, self.new_size)
            msk = cv2.resize(msk, self.new_size)


        if self.mask_half:
            msk = cv2.resize(msk, (msk.shape[1] // 2, msk.shape[0] // 2))

        msk = (msk > 127)
        msk = msk[:, :, np.newaxis]

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        pixel_size /= 10.0
        pixel_size = torch.from_numpy(np.array([pixel_size])).float()

        lbl_bin = np.zeros((5,), dtype=int)
        lbl_bin[organs.index(r['organ'])] = 1
        lbl_bin = torch.from_numpy(lbl_bin).float()

        sample = {'img': img, 'msk': msk, 'id': str(r['id']), 'organ': r['organ'], 'pixel_size': pixel_size, 'lbl': organs.index(r['organ']), 'lbl_bin': lbl_bin, 'tta': tta, 'data_source': r['data_source']}

        return sample



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

        img0 = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_COLOR)

        orig_shape = img0.shape

        sample = {'id': r['id'], 'organ': r['organ'], 'orig_h': orig_shape[0], 'orig_w': orig_shape[1]}

        for i in range(len(self.new_size)):

            img = cv2.resize(img0, self.new_size[i])

            img = preprocess_inputs(img)
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            sample['img{}'.format(i)] = img

        return sample




class TestExternalDataset(Dataset):
    def __init__(self, files, data_dir='external_images', new_size=None):
        super().__init__()
        self.files = files
        self.data_dir = data_dir
        self.new_size = new_size

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        fn = self.files[idx]

        img0 = cv2.imread(path.join(self.data_dir, fn), cv2.IMREAD_COLOR)

        orig_shape = img0.shape

        sample = {'id': fn, 'orig_h': orig_shape[0], 'orig_w': orig_shape[1]}

        for i in range(len(self.new_size)):

            img = cv2.resize(img0, self.new_size[i])

            img = preprocess_inputs(img)
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            sample['img{}'.format(i)] = img

        return sample