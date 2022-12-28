# import os
# import random
import numpy as np
# import cv2
# from skimage import measure
import pandas as pd


# import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')


df['fold'] = (df['sex'] == 'Male') * 100 + df['age']
_sorts = []
for o in df['organ'].unique():
    _sorts.append(df[df['organ'] == o].sort_values(by='fold').index.values)

for _sort in _sorts:
    for fold in range(5):
        df.loc[_sort[fold::5], 'fold'] = fold

df['fold'] = df['fold'].astype('int')

# for fold in range(5):
#     print(fold, sns.displot(df[df['fold'] == fold]['sex']))

df.to_csv('folds.csv', index=False)