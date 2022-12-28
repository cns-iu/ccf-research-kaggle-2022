import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
from ipywidgets import interact
from PIL import Image
from math import ceil
from bs4 import BeautifulSoup
import requests
from io import StringIO, BytesIO
from tqdm.auto import tqdm

out_dir = 'hpa_images_extra'
os.makedirs(out_dir, exist_ok=True)



r = requests.get("https://www.proteinatlas.org/api/search_download.php?search=&columns=g&compress=no&format=tsv")
string = StringIO(r.text)
df = pd.read_csv(string, sep="\t")
df.head()




organs = ['lung', 'prostate' 'spleen', 'kidney', 'colon']

metadata = []

pbar = tqdm(df.index[:200])

img_idx = 0
for idx in pbar:
    # continue
    ens = df.loc[idx, "Ensembl"]
    gene = df.loc[idx, "Gene"]
    for organ in organs:
        url = f"https://www.proteinatlas.org/{ens}-{gene}/tissue/{organ}"
        
        r = requests.get(url)
        images = BeautifulSoup(r.text, 'html.parser').findAll('img')
        links = ['https:' + img['src'].replace('_medium', '') for img in images if img['src'].startswith('//images.proteinatlas.org')]
        
        for link in links:
#             img_name = link.split('/')[-1]
            img_name = f"{organ}_{img_idx}.jpg"
            img_idx += 1
            r = requests.get(link)
            img = Image.open(BytesIO(r.content))
            img.save(os.path.join(out_dir, img_name))
            metadata.append([img_name, organ])
            
            pbar.set_description(f"{len(metadata):>5d} imgs saved")
            pbar.refresh()



pd.DataFrame(metadata).to_csv("./hpa_unlabeled.csv")