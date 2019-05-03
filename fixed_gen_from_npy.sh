#!/bin/bash

m=50
alpha=0.0
base_filter_num=16
train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
seed=0
fixed_gen_as_negative=True
fixed_negatives_npy='./pictures/fashion_mnist_vs_mnist_aug=False_alpha=0.5_m=50_seed=0_save_fixed_gen=True/fashion_mnist_vs_mnist_aug=False_alpha=0.5_m=50_seed=0_save_fixed_gen=True_fashion_mnist_fixed_gen_epoch500_iter500000.npy'
name="${test_dataset_a}_vs_${test_dataset_b}_aug=${use_augmented_variance_loss}_alpha=${alpha}_m=${m}_seed=${seed}_fixed_gen_as_negative=${fixed_gen_as_negative}_fixed_negatives_from_npy"
CUDA_VISIBLE_DEVICES=1 python main.py --name "fixed_negatives_from_npy_fashion_mnist_vs_mnist" --seed ${seed} --fixed_gen_as_negative ${fixed_gen_as_negative} --fixed_negatives_npy ${fixed_negatives_npy} --color False --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture dcgan_univ --oneclass_eval True --normal_class -1 --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/${name}.cout 2> pictures/${name}.cerr
