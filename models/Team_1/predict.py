"""
python predict.py

Read test data from TEST_DATA_CLEAN_PATH (specified in SETTINGS.json)
Load your model from MODEL_DIR (specified in SETTINGS.json)
Use your model to make predictions on new samples
Save your predictions to SUBMISSION_DIR (specified in SETTINGS.json)
"""

# import glob
import json
import os
import time

import numpy as np
import pandas as pd

from mmseg.apis import init_segmentor, inference_segmentor


def mask2rle(img):
    """
    img: numpy arrray, 1=mask, 0=background
    Returns run length as string formatted
    """
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape, val=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    val: val of the mask
    Returns numpy array, val - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = val
    return img.reshape(shape).T


def main():
    json_path='./SETTINGS.json'

    with open(json_path, 'r') as f:
        config=json.load(f)
    TEST_DATA_IMAGE_PATH=config['TEST_DATA_IMAGE_PATH']
    TEST_DATA_CSV_PATH=config['TEST_DATA_CSV_PATH']
    USE_MODEL=config['USE_MODEL']
    SUBMISSION_DIR=config['SUBMISSION_DIR']

    MODELS=[]
    MODEL_CONFIGS=[]
    assert isinstance(USE_MODEL, list), 'USE_MODEL must be list'
    for i in USE_MODEL:
        assert isinstance(i, int), 'Values of MODELS must be int'

        model_path=config[f'MODEL_{i}']
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f)
        else:
            MODELS.append(model_path)

        model_config_path=config[f'MODEL_CONFIG_{i}']
        if not os.path.isfile(model_config_path):
            raise FileNotFoundError(model_config_path)
        else:
            MODEL_CONFIGS.append(model_config_path)

    df=pd.read_csv(TEST_DATA_CSV_PATH)
    print('**all files:', len(df))

    models=[]
    for checkpoint, config_file in zip(MODELS, MODEL_CONFIGS):
        model=init_segmentor(config_file, checkpoint, device='cuda:0')
        models.append(model)

    t=time.time()
    cnt=0
    submission=[]
    for i in df.id:
        organ=df[df.id==i]['organ'].iloc[0]
        
        if USE_ORGAN[organ]:
            img_path=os.path.join(TEST_DATA_IMAGE_PATH, str(i) + IMG_SUFFIX)

            preds=[inference_segmentor(model, img_path)[0] for model in models]

            preds=np.array(preds)
            #print('preds_shape:', preds.shape)
            averaged_pred=np.mean(preds, axis=0)
            averaged_pred=np.where(averaged_pred[1]>=THRESHOLDS[organ], 1, 0)
            #print('avraged_pred shape:', averaged_pred.shape)
            
            rle=mask2rle(averaged_pred)
            submission.append([i, rle])
            print(f'processed: {img_path} | organ:{organ} | threshold:{THRESHOLDS[organ]}')
        else:
            submission.append([i, ''])
        
        cnt+=1
        if (cnt+1)%10==0:
            print('-----------------')
            print(f'{cnt+1}/{len(df.id)}')
        
        
    print(f'Processed {cnt} items.')
    elapsed=time.time()-t
    print(f'elapsed:{elapsed:.1f}s')
    
    if not os.path.isdir(SUBMISSION_DIR):
        os.mkdir(SUBMISSION_DIR)
    save_path=os.path.join(SUBMISSION_DIR, 'submission.csv')
    
    if os.path.isfile(save_path):
        os.remove(save_path)

    df_sub=pd.DataFrame(submission, columns=['id', 'rle'])
    df_sub.to_csv(save_path, index=False)


if __name__=='__main__':
    IMG_SUFFIX='.tiff'

    THRESHOLDS={
        'kidney':0.3,
        'largeintestine':0.2,
        'lung':0.6,
        'prostate':0.3,
        'spleen':0.5,
    }

    USE_ORGAN={
        'kidney':True,
        'largeintestine':True,
        'lung':True,
        'prostate':True,
        'spleen':True,
    }

    main()


