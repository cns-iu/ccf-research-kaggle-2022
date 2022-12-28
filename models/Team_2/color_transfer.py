from os import path, makedirs

import numpy as np
import random

import cv2

import pandas as pd

# from utils import *

# import matplotlib.pyplot as plt
# import seaborn as sns

import skimage.io

from tqdm import tqdm




organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

hubmap_pix_sizes = {
            'kidney': 0.5,
            'prostate': 6.263,
            'largeintestine': 0.229,
            'spleen': 0.4945,
            'lung': 0.7562
        }

df = pd.read_csv('folds.csv')

import staintools


makedirs('train_color_transfered', exist_ok=True)


_idx = 0
for target_file in ["test_images/10078.tiff", "PAS-staining.jpg", "HE-Stain-10x.jpg"]:

    print(target_file)

    target = staintools.read_image(target_file)

    # Standardize brightness (optional, can improve the tissue mask calculation)
    target = staintools.LuminosityStandardizer.standardize(target)

    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)

    for _, r in tqdm(df.iterrows()):
        to_transform = staintools.read_image(path.join("train_images", "{}.tiff".format(r['id'])))
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        transformed1 = normalizer.transform(to_transform)

        transformed1 = transformed1[:, :, ::-1]
        cv2.imwrite(path.join('train_color_transfered', '{}_{}.png'.format(r['id'], _idx)), transformed1, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        

    _idx += 1
