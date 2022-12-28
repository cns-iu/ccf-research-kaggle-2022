optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(warmup='linear', warmup_iters=500, by_epoch=False, 
                 policy='poly', power=0.9, min_lr=0.001)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=35, metric='mIoU', pre_eval=False)