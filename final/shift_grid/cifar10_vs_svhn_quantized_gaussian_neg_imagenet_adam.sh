#!/bin/bash

m=30
beta=1.0
alpha_reconstructed=0.0
alpha_generated=1.0
alpha_neg=0.0
neg_dataset=None
base_filter_num=32 # Here it is irrelevant
train_dataset='cifar10'
test_dataset_a='cifar10'
test_dataset_b='svhn_cropped'
use_augmented_variance_loss=False
seed=0
normal_class=-1
model_architecture='dcgan_univ'
lr_schedule='constant'
obs_noise_model='gaussian'
initial_log_gamma=0.0
reg_lambda=1.0
neg_prior=True
tags="junk,adam,neg_adv,bn,${obs_noise_model},${model_architecture},${test_dataset_a},${test_dataset_b}"
d=$(date +%Y-%m-%d-%H:%M:%S)
neg_prior_mean_coeff=25
trained_gamma=False
optimizer='adam'

for seed in 1 2 3 4 5
do
    name_prefix="${d}_FINAL_neg_adv_quantized_gaussian_1_bn"
    name="${name_prefix}_${initial_log_gamma}_${test_dataset_a}_vs_${test_dataset_b}_${model_architecture}_alpha_gen=${alpha_generated}_alpha_neg=${alpha_neg}_neg_dataset=${neg_dataset}_nc=${normal_class}_seed=${seed}"
    CUDA_VISIBLE_DEVICES=4 python main.py --tags ${tags} --neg_prior ${neg_prior} --neg_prior_mean_coeff ${neg_prior_mean_coeff} --initial_log_gamma ${initial_log_gamma} --trained_gamma ${trained_gamma} --optimizer ${optimizer} --add_obs_noise True --reg_lambda ${reg_lambda} --obs_noise_model ${obs_noise_model} --neg_train_size 60000 --neg_test_size 10000 --alpha_neg ${alpha_neg} --neg_dataset ${neg_dataset} --name "${name_prefix}_${test_dataset_a}_vs_${test_dataset_b}_${model_architecture}" --seed ${seed} --color True --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture ${model_architecture} --oneclass_eval True --normal_class ${normal_class} --gcnorm None --dataset ${train_dataset} --shape 32,32 --train_size 50000 --test_size 10000 --m ${m} --alpha_reconstructed ${alpha_reconstructed} --alpha_generated ${alpha_generated} --beta ${beta} --latent_dim 100 --batch_size 50 --lr 0.0001 --nb_epoch 100 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.00000 --generator_use_bn True --generator_wd 0.00000 --frequency 1000 --verbose 2 --resnet_wideness 1 --lr_schedule ${lr_schedule} > pictures/${name}.cout 2> pictures/${name}.cerr
done
