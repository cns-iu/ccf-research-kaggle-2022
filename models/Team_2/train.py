from operator import le
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
import random

from contextlib import suppress

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from torch.optim import SGD

from torch.optim.adamw import AdamW
from adan import Adan

from losses import iou_round, dice_round, ComboLoss

from sklearn.metrics import mean_squared_error, log_loss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from models import Timm_Unet

from Dataset import TrainDataset, ValDataset
from utils import *

from timm.utils.distributed import distribute_bn

from ddp_utils import all_gather, reduce_tensor

import timm

from torch.utils.tensorboard import SummaryWriter

# import warnings
# warnings.filterwarnings("ignore")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amp', default=True, type=bool)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--encoder", default='coat_lite_medium')
parser.add_argument("--checkpoint", default='coat_lite_medium_1472_e42')
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument("--checkpoint_path", default='')
parser.add_argument("--continue_best", default=False, type=bool)
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--img_size", default=1472, type=int)

args, unknown = parser.parse_known_args()


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

local_rank = 0
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
args.local_rank = local_rank


df = pd.read_csv('folds.csv')

df_ext = pd.read_csv('external.csv')
df = df.append(df_ext)
df = df.reset_index(drop=True)

df['id'] = df['id'].astype('str')


organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

masks_dir = 'masks'
data_dir = 'train_images'

models_folder = 'weights'



def validate(model, val_data_loader, current_epoch, amp_autocast=suppress):
    metrics = [[] for i in range(2)]
    cls_pred = []
    cls_truth = []
    ids = []
    tta = []
    data_source = []

    if args.local_rank == 0:
        iterator = tqdm(val_data_loader)
    else:
        iterator = val_data_loader

    with torch.no_grad():
        for i, sample in enumerate(iterator):
            with amp_autocast():
                _organs = sample['organ']

                imgs = sample['img'].cuda(non_blocking=True)
                otps = sample['msk'].cpu().numpy()
                lbls = sample["lbl"].cpu().numpy()
                pix_szs = sample["pixel_size"].cpu().numpy()
                _tta = sample["tta"].cpu().numpy()
                _data_source = sample["data_source"]
                
                res, res_cls, res_pix = model(imgs)
            # break
                probs = torch.sigmoid(res)
                pred = probs.cpu().numpy()
                
                res_cls = torch.softmax(res_cls, dim=1).cpu().numpy()
                res_pix = res_pix.cpu().numpy()
                
                for j in range(otps.shape[0]):
                    ids.append(organs.index(_organs[j]))
                    tta.append(_tta[j])
                    data_source.append(_data_source[j])

                    for l in range(1):
                        _truth = otps[j, l]
                        _pred = pred[j, l] > 0.5

                        _dice = dice(_truth, _pred)
                        
                        metrics[l].append(_dice)

                    cls_pred.append(res_cls[j])
                    cls_truth.append(lbls[j])

                    
                        
                metrics[1].append(mean_squared_error(pix_szs.astype(np.float64), res_pix.astype(np.float64)))

    metrics = [np.asarray(x) for x in metrics]
    ids = np.asarray(ids)
    cls_pred = np.asarray(cls_pred)
    cls_truth = np.asarray(cls_truth)
    tta = np.asarray(tta)
    data_source = np.asarray(data_source)


    if args.distributed:
        metrics = [np.concatenate(all_gather(x)) for x in metrics]
        ids = np.concatenate(all_gather(ids))
        cls_pred = np.concatenate(all_gather(cls_pred))
        cls_truth = np.concatenate(all_gather(cls_truth))
        tta = np.concatenate(all_gather(tta))
        data_source = np.concatenate(all_gather(data_source))
        torch.cuda.synchronize()

    d0 = np.mean(metrics[0])

    pix_mse = np.mean(metrics[1])

    cce = log_loss(cls_truth, cls_pred)

    _dice_mean = 0
    for i in range(len(organs)):
        _sc = np.mean(metrics[0][ids == i])
        _dice_mean += _sc
        if args.local_rank == 0:
            writer.add_scalar("Dice/Val " + organs[i], _sc, current_epoch)
            print("Val {}: {}".format(organs[i], _sc))

    _dice_mean /= len(organs)

    tta_sc = []
    for i in range(4):
        _sc = np.mean(metrics[0][tta == i])
        tta_sc.append(_sc)

    ext_sc = np.mean(metrics[0][data_source == 'external'])

    if args.local_rank == 0:
        print("Val Dice: {} av: {} ext_sc: {} av_tta: {} pix_mse: {} cce: {} Len: {}".format(_dice_mean, d0, ext_sc, tta_sc, pix_mse, cce, len(metrics[0])))
        writer.add_scalar("Dice/Val", _dice_mean, current_epoch)
        writer.add_scalar("Dice/Val_ext", ext_sc, current_epoch)
        writer.add_scalar("Av_Dice/Val", d0, current_epoch)
        writer.add_scalar("Av_Dice_tta0/Val", tta_sc[0], current_epoch)
        writer.add_scalar("Loss mse/Val", pix_mse, current_epoch)
        writer.add_scalar("Loss cce/Val", cce, current_epoch)


    _dice_mean = _dice_mean * 0.5 + ext_sc * 0.5

    return _dice_mean


def evaluate_val(val_data_loader, best_score, model, snapshot_name, current_epoch, amp_autocast=suppress):
    model.eval()
    _sc = validate(model, val_data_loader, current_epoch, amp_autocast)

    if args.local_rank == 0:
        if _sc > best_score:
            if args.distributed:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))
            else:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))

            best_score = _sc
        print("Val score: {}\tbest_score: {}".format(_sc, best_score))
    return best_score, _sc



def train_epoch(current_epoch, combo_loss, ce_loss, mse_loss, model, optimizer, scaler, train_data_loader, amp_autocast=suppress):
    losses = [AverageMeter() for i in range(10)]
    metrics = [AverageMeter() for i in range(10)]
    
    if args.local_rank == 0:
        iterator = tqdm(train_data_loader)
    else:
        iterator = train_data_loader

    _lr = optimizer.param_groups[0]['lr']

    model.train()

    for i, sample in enumerate(iterator):
        with amp_autocast():
            imgs = sample["img"].cuda(non_blocking=True)
            otps = sample["msk"].cuda(non_blocking=True)
            lbls = sample["lbl"].cuda(non_blocking=True)
            lbls_bin = sample["lbl_bin"].cuda(non_blocking=True)
            pix_szs = sample["pixel_size"].cuda(non_blocking=True)

        # break
            res, res_cls, res_pix = model(imgs)

            loss0 = combo_loss(res, otps)

            cls_loss = ce_loss(res_cls, lbls_bin)
            pix_loss = mse_loss(res_pix, pix_szs)

            loss = loss0 + 0.1 * cls_loss + 0.1 * pix_loss

            if current_epoch < start_epoch + 1:
                loss = loss * 0.05 # warm-up

        _dices = []
        with torch.no_grad():
            for _i in range(1): #otps.shape[1]
                _probs = torch.sigmoid(res[:, _i, ...])
                dice_sc = 1 - dice_round(_probs, otps[:, _i, ...])
                _dices.append(dice_sc)
            del _probs

        if args.distributed:
            reduced_loss = [reduce_tensor(x.data) for x in [loss, loss0, cls_loss, pix_loss]]
            reduced_sc = [reduce_tensor(x) for x in _dices]
        else:
            reduced_loss = [x.data for x in [loss, loss0, cls_loss, pix_loss]]
            reduced_sc = _dices

        for _i in range(len(reduced_loss)):
            losses[_i].update(to_python_float(reduced_loss[_i]), imgs.size(0))
        for _i in range(len(reduced_sc)):
            metrics[_i].update(reduced_sc[_i], imgs.size(0)) 

        if args.local_rank == 0:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss: {:.4f}  ({:.4f}) combo: {:.4f}  ({:.4f}) cce: {:.4f}  ({:.4f}) mse: {:.4f}  ({:.4f}) dices: {:.4f} ({:.4f})".format(
                    current_epoch, _lr, losses[0].val, losses[0].avg, losses[1].val, losses[1].avg, losses[2].val, losses[2].avg, losses[3].val, losses[3].avg, metrics[0].val, metrics[0].avg))


        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            optimizer.step()

        torch.cuda.synchronize()

    if args.local_rank == 0:
        _dice = metrics[0].avg
        writer.add_scalar("Loss/train", losses[1].avg, current_epoch)
        writer.add_scalar("Dice/train", _dice, current_epoch)
        writer.add_scalar("Loss cce/train", losses[2].avg, current_epoch)
        writer.add_scalar("Loss mse/train", losses[3].avg, current_epoch)
        writer.add_scalar("lr", _lr, current_epoch)

        print("epoch: {}; lr {:.7f}; Loss {:.4f} combo: {:.4f} cce: {:.4f} mse: {:.4f} dices: {:.4f};".format(
                    current_epoch, _lr, losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg, _dice))


start_epoch = 0
            

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(models_folder, exist_ok=True)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()


    fold = args.fold
    
    train_df = df[df['fold'] != fold].copy()
    train_df.reset_index(drop=True, inplace=True)
    
    val_df = df[df['fold'] == fold].copy()
    val_df.reset_index(drop=True, inplace=True)


    if args.local_rank == 0:
        writer = SummaryWriter(comment='__{}_{}'.format(args.checkpoint, fold))
        print(args)
        
    cudnn.benchmark = True

    batch_size = args.batch_size
    val_batch = args.batch_size

    best_snapshot_name = '{}_{}_best'.format(args.checkpoint, fold)
    last_snapshot_name = '{}_{}_last'.format(args.checkpoint, fold)



    new_size = (args.img_size, args.img_size) 

    data_train = TrainDataset(train_df, data_dir=data_dir, masks_dir=masks_dir, aug=True, new_size=new_size, epoch_size=1000)
    data_val = ValDataset(val_df, data_dir=data_dir, masks_dir=masks_dir, new_size=new_size)


    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val)


    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=12, shuffle=False, pin_memory=False, sampler=train_sampler, drop_last=True) #shuffle=(train_sampler is None)
    val_data_loader = DataLoader(data_val, batch_size=val_batch, num_workers=12, shuffle=False, pin_memory=False, sampler=val_sampler)


    model = Timm_Unet(name=args.encoder, pretrained=args.pretrained)


    if args.distributed:
        model = timm.models.layers.convert_sync_batchnorm(model)

    model = model.cuda()

    params = model.parameters()
    
    lr = 1e-4
    if args.continue_best:
        lr = 2e-5
    
    optimizer = Adan(params, lr=lr) 


    if args.continue_best:
        snap_to_load =  best_snapshot_name.format(fold)
        if path.exists(path.join(models_folder, snap_to_load)):
            if args.local_rank == 0:
                print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            model.load_state_dict(loaded_dict)
            if args.local_rank == 0:
                print("loaded checkpoint '{}' (epoch {}, best_score {})"
                    .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        start_epoch = checkpoint['epoch'] + 1


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
            output_device=args.local_rank)


    loss_scaler = None
    amp_autocast = suppress
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
        amp_autocast = torch.cuda.amp.autocast

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=9, verbose=False, threshold=0.00001, threshold_mode='abs', cooldown=0, min_lr=1e-06, eps=1e-06)

    combo_loss = ComboLoss({'dice': 1.0, 'bce': 0.5}, per_image=True).cuda() #
    ce_loss = nn.BCEWithLogitsLoss().cuda()
    mse_loss = nn.MSELoss().cuda()


    best_score = 0
    for epoch in range(start_epoch, args.epoches):
        torch.cuda.empty_cache()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(epoch, combo_loss, ce_loss, mse_loss, model, optimizer, loss_scaler, train_data_loader, amp_autocast)

        if args.distributed:
            distribute_bn(model, args.world_size, True)

        torch.cuda.empty_cache()

        best_score, _sc = evaluate_val(val_data_loader, best_score, model, best_snapshot_name, epoch, amp_autocast)
        scheduler.step(_sc)
        
        torch.cuda.empty_cache()

        # if args.local_rank == 0:
        #     writer.flush()

        #     if args.distributed:
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'state_dict': model.module.state_dict(),
        #             'best_score': best_score,
        #         }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
        #     else:
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'best_score': best_score,
        #         }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
    
    torch.cuda.empty_cache()
    if args.distributed:
        torch.cuda.synchronize()

    del model

    elapsed = timeit.default_timer() - t0
    if args.local_rank == 0:
        writer.close()
        print('Time: {:.3f} min'.format(elapsed / 60))