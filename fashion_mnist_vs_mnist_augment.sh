#!/bin/bash

m=30
alpha_reconstructed=0.1
alpha_generated=0.1
base_filter_num=16
train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
seed=0
reg_lambda=10.0

augment=True

name="${test_dataset_a}_vs_${test_dataset_b}_aug=${use_augmented_variance_loss}_reg_lambda=${reg_lambda}_alpha_reconst=${alpha_reconstructed}_alpha_gen=${alpha_generated}_m=${m}_seed=${seed}_augment=${augment}"
CUDA_VISIBLE_DEVICES=2 python main.py --name "fashion_mnist_vs_mnist_augment" --augment ${augment} --reg_lambda ${reg_lambda} --seed ${seed} --color False --use_augmented_variance_loss ${use_augmented_variance_loss} --test_dataset_b ${test_dataset_b} --test_dataset_a ${test_dataset_a} --model_architecture dcgan_univ --oneclass_eval True --normal_class -1 --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha_reconstructed ${alpha_reconstructed} --alpha_generated ${alpha_generated} --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/${name}.cout 2> pictures/${name}.cerr
