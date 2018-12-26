#!/bin/bash

for seed in {0..9}
do
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py --seed $seed --model_architecture deepsvdd --oneclass_eval True --normal_class 2 --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m 50 --alpha 0.25 --beta 1.0 --latent_dim 128 --batch_size 50 --lr 0.00002 --nb_epoch 100 --prefix pictures/bird_nbepoch=100/cifar10_seed=$seed\_class=2_m=50_alpha=0.25_beta=1.0_dim=128_bs=50_wd=0.000001 --model_path  pictures/bird_nbepoch=100/cifar10_seed=$seed\_class=2_m=50_alpha=0.25_beta=1.0_dim=128_bs=50_wd=0.000001/model --save_latent True --base_filter_num 16 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > cifar10_bird_epoch=100_seed=$seed.cout 2> cifar10_bird_epoch=100_seed=$seed.cerr
done