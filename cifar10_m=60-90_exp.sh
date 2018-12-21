#!/bin/bash

for seed in {0..9}
do
    for m in {60..90..10}
    do
        for class in {0..9}
        do
            CUDA_VISIBLE_DEVICES=1 python3.6 main.py --seed $seed --model_architecture deepsvdd --oneclass_eval True --normal_class $class --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m $m --alpha 0.25 --beta 1.0 --latent_dim 128 --batch_size 50 --lr 0.00002 --nb_epoch 40 --prefix pictures/cifar10_seed=$seed\_class=$class\_m=$m\_alpha=0.25_beta=1.0_dim=128_bs=50_wd=0.000001 --model_path  pictures/cifar10_seed=$seed\_class=$class\_m=$m\_alpha=0.25_beta=1.0_dim=128_bs=50_wd=0.000001/model --save_latent True --base_filter_num 16 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > cifar10_seed=$seed\_class=$class\_m=$m.cout 2> cifar10_seed=$seed\_class=$class\_m=$m.cerr
        done
    done
done