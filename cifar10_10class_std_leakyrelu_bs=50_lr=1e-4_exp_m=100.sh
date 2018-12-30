#!/bin/bash
m=100
lrschedule=exponential
lr=0.0001
bs=50
seed=0
for class in {0..9}
do
	name=cifar10_class=${class}_seed=${seed}_m=${m}_alpha=0.25_beta=1.0_dim=128_bs=${bs}_wd=0.000001_std_leakyrelu_lr=${lr}_${lrschedule}
    CUDA_VISIBLE_DEVICES=2 python3.6 main.py --lr_schedule $lrschedule --gcnorm None --seed 0 --model_architecture deepsvdd --oneclass_eval True --normal_class $class --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m $m --alpha 0.25 --beta 1.0 --latent_dim 128 --batch_size $bs --lr $lr --nb_epoch 100 --prefix pictures/$name --model_path  pictures/$name/model --save_latent True --base_filter_num 32 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > $name.cout 2> $name.cerr
done