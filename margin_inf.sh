#!/bin/bash

train_dataset='fashion_mnist'
test_dataset_a='fashion_mnist'
test_dataset_b='mnist'
use_augmented_variance_loss=False
base_filter_num=16
seed=0

margin_inf=True

for m in 0
do
    for alpha in 0.5
    do
        for reg_lambda in 1.0
        do
            name="${test_dataset_a}_vs_${test_dataset_b}_aug=${use_augmented_variance_loss}_alpha=${alpha}_m=${m}_reg_lambda=${reg_lambda}_margin_inf=${margin_inf}_seed=${seed}"
            CUDA_VISIBLE_DEVICES=1 python main.py --name "margin_inf_fashion_mnist_vs_mnist" --seed ${seed} --color False --margin_inf ${margin_inf} --use_augmented_variance_loss ${use_augmented_variance_loss} --reg_lambda ${reg_lambda} --test_dataset_a ${test_dataset_a} --test_dataset_b ${test_dataset_b} --model_architecture dcgan_univ --oneclass_eval True --normal_class -1 --gcnorm None --dataset ${train_dataset} --shape 28,28 --train_size 50000 --test_size 10000 --m ${m} --alpha ${alpha} --beta 1.0 --latent_dim 10 --batch_size 50 --lr 0.0001 --nb_epoch 500 --prefix pictures/${name} --model_path pictures/${name}/model --save_latent True --base_filter_num ${base_filter_num} --encoder_use_bn True --encoder_wd 0.000001 --generator_use_bn True --generator_wd 0.000001 --frequency 100 --verbose 2 --resnet_wideness 1 > pictures/${name}.cout 2> pictures/${name}.cerr
        done
    done
done
