#!/bin/bash

m=30
alpha=0.25

CUDA_VISIBLE_DEVICES=0 python3.6 main.py --seed 0 --model_architecture dcgan --oneclass_eval True --normal_class -1 --gcnorm None --dataset cifar10 --shape 32,32 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta 1.0 --latent_dim 128 --batch_size 200 --lr 0.0001 --nb_epoch 200 --prefix pictures/cifar10_vs_svhn_m=${m}_alpha=${alpha} --model_path pictures/cifar10_vs_svhn_m=${m}_alpha=${alpha}/model --save_latent True --base_filter_num 16 --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/cifar10_vs_svhn_m=${m}_alpha=${alpha}.cout 2> pictures/cifar10_vs_svhn_m=${m}_alpha=${alpha}.cerr
