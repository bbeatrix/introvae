#!/bin/bash

m=30
beta=1.0
alpha_reconstructed=0.0
alpha_generated=0.0
alpha_neg=0.0
neg_dataset=None
base_filter_num=32
train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
seed=0
normal_class=-1
model_architecture='dcgan_univ'
lr_schedule='constant'

for seed in 0 1 2 3 4
do
   name="FIN_baseline_${test_dataset_a}_vs_${test_dataset_b}_${model_architecture}_alpha_reconstructed=${alpha_reconstructed}_alpha_generated=${alpha_generated}_alpha_neg=${alpha_neg}_neg_dataset=${neg_dataset}_m=${m}_beta=${beta}_nc=${normal_class}_seed=${seed}"
   CUDA_VISIBLE_DEVICES=2 python main.py --reg_lambda 1.0 --name "FIN_baseline_${test_dataset_a}_vs_${test_dataset_b}_${model_architecture}" --seed ${seed} --color False --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture ${model_architecture} --oneclass_eval True --normal_class ${normal_class} --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha_reconstructed ${alpha_reconstructed} --alpha_generated ${alpha_generated} --beta ${beta} --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 150 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 1000 --verbose 2 --resnet_wideness 1 --lr_schedule ${lr_schedule} > pictures/${name}.cout 2> pictures/${name}.cerr
done
