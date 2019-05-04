#!/bin/bash

m=50
alpha=0.5
base_filter_num=16
train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
seed=0
save_fixed_generated=True
name="${test_dataset_a}_vs_${test_dataset_b}_aug=${use_augmented_variance_loss}_alpha=${alpha}_m=${m}_seed=${seed}_save_fixed_generated=${save_fixed_generated}"
CUDA_VISIBLE_DEVICES=0 python main.py --name "save_fixed_gen_fashion_mnist_vs_mnist" --seed ${seed} --save_fixed_gen ${save_fixed_generated} --color False --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture dcgan_univ --oneclass_eval True --normal_class -1 --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/${name}.cout 2> pictures/${name}.cerr
