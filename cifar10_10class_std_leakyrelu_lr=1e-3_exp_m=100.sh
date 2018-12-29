#!/bin/bash
m=100
lrschedule=exponential
lr=0.001
for class in {0..9}
do
    CUDA_VISIBLE_DEVICES=1 python3.6 main.py --lr_schedule $lrschedule --gcnorm None --seed 0 --model_architecture deepsvdd --oneclass_eval True --normal_class $class --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m $m --alpha 0.25 --beta 1.0 --latent_dim 128 --batch_size 200 --lr $lr --nb_epoch 100 --prefix pictures/cifar10_std_leakyrelu_lr=${lr}_${lrschedule}_m=$m/cifar10_class=$class\_seed=0_m=$m\_alpha=0.25_beta=1.0_dim=128_bs=200_wd=0.000001_std_leakyrelu_lr=${lr}_${lrschedule} --model_path  pictures/cifar10_std_leakyrelu_lr=${lr}_${lrschedule}/cifar10_class=$class\_seed=0_m=$m\_alpha=0.25_beta=1.0_dim=128_bs=200_wd=0.000001_std_leakyrelu_lr=${lr}_${lrschedule}/model --save_latent True --base_filter_num 32 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > cifar10_class=$class\_std_leakyrelu_lr=${lr}_${lrschedule}_m=$m.cout 2> cifar10_class=$class\_std_leakyrelu_lr=${lr}_${lrschedule}_m=$m.cerr
done