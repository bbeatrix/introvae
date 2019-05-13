#!/bin/bash

m=40
beta=0.1
#alpha=0.25
alpha_reconstructed=1.0
alpha_generated=1.0
base_filter_num=32
train_dataset='cifar10'
test_dataset_a='cifar10'
test_dataset_b='cifar10'
use_augmented_variance_loss=False
seed=0
normal_class=4
model_architecture='dcgan_univ'
lr_schedule='constant'

for seed in 0 1 2 3 4
do
   name="final_nowd_nobn_normed_${test_dataset_a}_vs_${test_dataset_b}_${model_architecture}_aug=${use_augmented_variance_loss}_alpha_reconstructed=${alpha_reconstructed}_alpha_generated=${alpha_generated}_m=${m}_beta=${beta}_nc=${normal_class}_seed=${seed}"
   CUDA_VISIBLE_DEVICES=4 python main.py --name "final_cifar10_vs_cifar10_${model_architecture}" --seed ${seed} --color True --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture ${model_architecture} --oneclass_eval True --normal_class ${normal_class} --gcnorm None --dataset ${train_dataset} --shape 32,32 --train_size 50000 --test_size 10000 --m ${m} --alpha_reconstructed ${alpha_reconstructed} --alpha_generated ${alpha_generated} --beta ${beta} --latent_dim 100 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn False --encoder_wd 0.00000 --generator_use_bn False --generator_wd 0.00000 --frequency 100 --verbose 2 --resnet_wideness 1 --lr_schedule ${lr_schedule} > pictures/${name}.cout 2> pictures/${name}.cerr
done