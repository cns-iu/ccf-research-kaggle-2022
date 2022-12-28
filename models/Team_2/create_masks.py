import os
import numpy as np
import cv2
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns

from tqdm import tqdm

df = pd.read_csv('folds.csv')

masks_dir = 'masks'

os.makedirs(masks_dir, exist_ok=True)


def rle2mask(mask_rle, shape):

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T


for _, r in tqdm(df.iterrows()):
    msk = rle2mask(r['rle'], (r['img_width'], r['img_height']))
    cv2.imwrite(os.path.join(masks_dir, '{}.png'.format(r['id'])), msk, [cv2.IMWRITE_PNG_COMPRESSION, 5])