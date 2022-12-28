log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

work_dir=''

from mmseg.apis import set_random_seed
set_random_seed(0, deterministic=False)

seed = 0
gpu_ids = range(1)
device='cuda'