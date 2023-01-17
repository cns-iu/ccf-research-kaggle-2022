import sys
sys.path.insert(0, '../../')
import warnings
warnings.simplefilter('ignore')

from utils import fix_seed, elapsed_time
from utils_data_generation import HuBMAPDataset, my_collate_fn, generate_data

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import os
from os.path import join as opj

import time
start = time.time()

VERSION = '01_01'
BASE_PATH = r'/N/slate/yashjain/kaggle_data_package/kaggle_data_multiftu/data/'
# dataset = "kidney"
dataset = "gftu" 

def get_config():
    config = {
        'VERSION':VERSION,
        'OUTPUT_PATH':f'./result/{VERSION}/',
        'INPUT_PATH':BASE_PATH +f'{dataset}_dataset/',
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'tile_size':1024,
        'batch_size':16,
        'shift_h':0,
        'shift_w':0,
    }
    return config
    
if __name__=='__main__':
    fix_seed(2020)
    config = get_config()
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']

    # import data
    train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
    if dataset == "gftu":
        train_df = train_df.rename(columns = {"rle":"encoding","filename":"id"})
        # train_df = train_df[train_df.id != "HandE_B005_CL_b_RGB_topright"]
    info_df  = pd.read_csv(opj(INPUT_PATH,'metadata.csv'))
    sub_df = pd.read_csv(opj(INPUT_PATH, 'test.csv'))

    print('train_df.shape = ', train_df.shape, "\n", train_df)
    print('info_df.shape  = ', info_df.shape)
    print('sub_df.shape = ', sub_df.shape)
    
    # generate training data
    data_list = []
    for idx,filename in enumerate(train_df['id'].values):
        print('idx = {}, {}'.format(idx,filename))
        ds = HuBMAPDataset(train_df, filename, config)
        #rasterio cannot be used with multiple workers
        dl = DataLoader(ds,batch_size=config['batch_size'],
                        num_workers=0,shuffle=False,pin_memory=True,
                        collate_fn=my_collate_fn)
        img_patches = []
        mask_patches = []

        for i, data in tqdm(enumerate(dl)):
            img_patch = data['img'].astype(np.uint8)
            mask_patch = data['mask'].astype(np.uint8)
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

        img_patches  = np.vstack(img_patches)
        mask_patches = np.vstack(mask_patches)

        # sort by number of masked pixels
        bs,sz,sz,c = img_patches.shape
        idxs = np.argsort(mask_patches.reshape(bs,-1).sum(axis=1))[::-1]
        img_patches  = img_patches[idxs].reshape(-1,sz,sz,c)
        mask_patches = mask_patches[idxs].reshape(-1,sz,sz,1)

        data = Parallel(n_jobs=-1)(delayed(generate_data)(filename, i, x, y, config) for i,(x,y) in enumerate(tqdm(zip(img_patches, mask_patches)))) 
        data_list.append(data)
    
    # save
    data_df = pd.concat([pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0).reset_index(drop=True)
    data_df.columns = ['filename_img', 'filename_rle', 'num_masked_pixels', 'ratio_masked_area', 'std_img']
    data_df.to_csv(opj(OUTPUT_PATH, 'data.csv'), index=False)

print ("Run time = ", elapsed_time(start))