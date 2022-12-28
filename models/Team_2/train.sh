mkdir -p logs

echo "Creating masks"
echo "python create_masks.py"

echo "Color Transfering for train images"
python color_transfer.py

echo "Training coat_lite_medium with image size 768"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_768_e40 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/coat_lite_medium_768_e40_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_768_e40 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/coat_lite_medium_768_e40_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_768_e40 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/coat_lite_medium_768_e40_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_768_e40 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/coat_lite_medium_768_e40_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_768_e40 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/coat_lite_medium_768_e40_4.txt

echo "Training coat_lite_medium with image size 1024"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1024_e41 --img_size=1024 --batch_size=5 --epoches=100 2>&1 | tee logs/coat_lite_medium_1024_e41_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1024_e41 --img_size=1024 --batch_size=5 --epoches=100 2>&1 | tee logs/coat_lite_medium_1024_e41_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1024_e41 --img_size=1024 --batch_size=5 --epoches=100 2>&1 | tee logs/coat_lite_medium_1024_e41_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1024_e41 --img_size=1024 --batch_size=5 --epoches=100 2>&1 | tee logs/coat_lite_medium_1024_e41_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1024_e41 --img_size=1024 --batch_size=5 --epoches=100 2>&1 | tee logs/coat_lite_medium_1024_e41_4.txt

echo "Training coat_lite_medium with image size 1472"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1472_e42 --img_size=1472 --batch_size=6 --epoches=100 2>&1 | tee logs/coat_lite_medium_1472_e42_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1472_e42 --img_size=1472 --batch_size=6 --epoches=100 2>&1 | tee logs/coat_lite_medium_1472_e42_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1472_e42 --img_size=1472 --batch_size=6 --epoches=100 2>&1 | tee logs/coat_lite_medium_1472_e42_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1472_e42 --img_size=1472 --batch_size=6 --epoches=100 2>&1 | tee logs/coat_lite_medium_1472_e42_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=coat_lite_medium --checkpoint=coat_lite_medium_1472_e42 --img_size=1472 --batch_size=6 --epoches=100 2>&1 | tee logs/coat_lite_medium_1472_e42_4.txt

echo "Training tf_efficientnet_b7_ns with image size 768"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_768_e34 --img_size=768 --batch_size=6 --epoches=170 2>&1 | tee logs/tf_efficientnet_b7_ns_768_e34_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_768_e34 --img_size=768 --batch_size=6 --epoches=170 2>&1 | tee logs/tf_efficientnet_b7_ns_768_e34_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_768_e34 --img_size=768 --batch_size=6 --epoches=170 2>&1 | tee logs/tf_efficientnet_b7_ns_768_e34_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_768_e34 --img_size=768 --batch_size=6 --epoches=170 2>&1 | tee logs/tf_efficientnet_b7_ns_768_e34_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_768_e34 --img_size=768 --batch_size=6 --epoches=170 2>&1 | tee logs/tf_efficientnet_b7_ns_768_e34_4.txt

echo "Training tf_efficientnet_b7_ns with image size 1024"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1024_e33 --img_size=1024 --batch_size=8 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1024_e33_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1024_e33 --img_size=1024 --batch_size=8 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1024_e33_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1024_e33 --img_size=1024 --batch_size=8 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1024_e33_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1024_e33 --img_size=1024 --batch_size=8 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1024_e33_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1024_e33 --img_size=1024 --batch_size=8 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1024_e33_4.txt

echo "Training tf_efficientnet_b7_ns with image size 1472"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1472_e35 --img_size=1472 --batch_size=3 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1472_e35_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1472_e35 --img_size=1472 --batch_size=3 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1472_e35_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1472_e35 --img_size=1472 --batch_size=3 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1472_e35_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1472_e35 --img_size=1472 --batch_size=3 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1472_e35_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnet_b7_ns --checkpoint=tf_efficientnet_b7_ns_1472_e35 --img_size=1472 --batch_size=3 --epoches=130 2>&1 | tee logs/tf_efficientnet_b7_ns_1472_e35_4.txt

echo "Training tf_efficientnetv2_l_in21ft1k with image size 768"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_768_e36 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_768_e36_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_768_e36 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_768_e36_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_768_e36 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_768_e36_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_768_e36 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_768_e36_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_768_e36 --img_size=768 --batch_size=6 --epoches=150 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_768_e36_4.txt

echo "Training tf_efficientnetv2_l_in21ft1k with image size 1024"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1024_e38 --img_size=1024 --batch_size=4 --epoches=130 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1024_e38_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1024_e38 --img_size=1024 --batch_size=4 --epoches=130 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1024_e38_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1024_e38 --img_size=1024 --batch_size=4 --epoches=130 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1024_e38_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1024_e38 --img_size=1024 --batch_size=4 --epoches=130 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1024_e38_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1024_e38 --img_size=1024 --batch_size=4 --epoches=130 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1024_e38_4.txt

echo "Training tf_efficientnetv2_l_in21ft1k with image size 1472"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1472_e39 --img_size=1472 --batch_size=4 --epoches=120 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1472_e39_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1472_e39 --img_size=1472 --batch_size=4 --epoches=120 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1472_e39_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1472_e39 --img_size=1472 --batch_size=4 --epoches=120 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1472_e39_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1472_e39 --img_size=1472 --batch_size=4 --epoches=120 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1472_e39_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=tf_efficientnetv2_l_in21ft1k --checkpoint=tf_efficientnetv2_l_in21ft1k_1472_e39 --img_size=1472 --batch_size=4 --epoches=120 2>&1 | tee logs/tf_efficientnetv2_l_in21ft1k_1472_e39_4.txt

echo "Training convnext_large_384_in22ft1k with image size 768"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_768_e37 --img_size=768 --batch_size=6 --epoches=100 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_768_e37 --img_size=768 --batch_size=6 --epoches=100 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_768_e37 --img_size=768 --batch_size=6 --epoches=100 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_768_e37 --img_size=768 --batch_size=6 --epoches=100 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_768_e37 --img_size=768 --batch_size=6 --epoches=100 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_4.txt

echo "Training convnext_large_384_in22ft1k with image size 1024"
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=0 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_1024_e32 --img_size=1024 --batch_size=4 --epoches=80 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_0.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=1 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_1024_e32 --img_size=1024 --batch_size=4 --epoches=80 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_1.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=2 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_1024_e32 --img_size=1024 --batch_size=4 --epoches=80 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_2.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=3 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_1024_e32 --img_size=1024 --batch_size=4 --epoches=80 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_3.txt
python -m torch.distributed.launch --nproc_per_node=2 train.py --fold=4 --encoder=convnext_large_384_in22ft1k --checkpoint=convnext_large_384_in22ft1k_1024_e32 --img_size=1024 --batch_size=4 --epoches=80 2>&1 | tee logs/convnext_large_384_in22ft1k_768_e37_4.txt

echo "All models trained!"